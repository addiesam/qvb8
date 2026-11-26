// Flash attention over a qvb8-banded KV cache for GPUs whose Triton has no
// tensor-core dot (Volta, sm_70): WMMA m16n16k16 fp16 -> fp32.
//
// One CTA = one kv head x one block of BM query rows (row r = token*G + g)
// x one split of the key tiles. Each BN-key tile is read as uint8 ids and
// de-quantised straight into shared memory (k = mid + step*(id - c), one
// (mid, step) per segment/head/channel), so the full-precision K/V of the
// context never exists anywhere: memory is O(BM x D) per CTA, independent of
// context length. Output is the split-local unnormalised accumulator + (m, l),
// combined on the host exactly like the Triton kernel it replaces.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

constexpr int BM = 64, BN = 32, NT = 256, NW = NT / 32;

template <int D>
struct Smem {
    static constexpr int LDQ = D + 8, LDK = D + 8, LDS = BN + 4, LDP = BN + 8;
    static constexpr size_t q = 0;
    static constexpr size_t kv = q + sizeof(half) * BM * LDQ;
    static constexpr size_t s = kv + sizeof(half) * BN * LDK;
    static constexpr size_t p = s + sizeof(float) * BM * LDS;
    static constexpr size_t scr = p + sizeof(half) * BM * LDP;      // 16x16 fp32 per warp
    static constexpr size_t st = scr + sizeof(float) * NW * 256;
    static constexpr size_t bytes = st + sizeof(float) * BM * 3;
};

template <typename T> struct Vec4;
template <> struct Vec4<uint8_t> { using type = uchar4; };
template <> struct Vec4<uint16_t> { using type = ushort4; };

template <int D, typename T>
__device__ __forceinline__ void load_tile(half* dst, const T* ids, long s_n, int start, int nlen,
                                          const float* mid, const float* step, float c) {
    constexpr int LDK = Smem<D>::LDK;
    for (int idx = threadIdx.x; idx < BN * D / 4; idx += NT) {
        int n = idx / (D / 4), c4 = (idx % (D / 4)) * 4;
        half2* out = reinterpret_cast<half2*>(dst + n * LDK + c4);
        if (n < nlen) {
            auto v = *reinterpret_cast<const typename Vec4<T>::type*>(ids + (long)(start + n) * s_n + c4);
            float4 md = *reinterpret_cast<const float4*>(mid + c4);
            float4 sp = *reinterpret_cast<const float4*>(step + c4);
            out[0] = __floats2half2_rn(md.x + sp.x * ((float)v.x - c), md.y + sp.y * ((float)v.y - c));
            out[1] = __floats2half2_rn(md.z + sp.z * ((float)v.z - c), md.w + sp.w * ((float)v.w - c));
        } else {
            out[0] = out[1] = __floats2half2_rn(0.f, 0.f);
        }
    }
}

// Register prefetch: the next tile's raw ids are fetched into registers while
// the current tile computes (one CTA per SM leaves no other warps to hide the
// global-memory latency), then de-quantised into shared memory.
template <int D> constexpr int kPer() { return BN * D / 4 / NT; }

template <int D, typename T>
__device__ __forceinline__ void fetch_ids(typename Vec4<T>::type (&r)[kPer<D>()], const T* ids,
                                          long s_n, int start, int nlen) {
#pragma unroll
    for (int i = 0; i < kPer<D>(); ++i) {
        int idx = threadIdx.x + i * NT;
        int n = idx / (D / 4), c4 = (idx % (D / 4)) * 4;
        if (n < nlen) r[i] = *reinterpret_cast<const typename Vec4<T>::type*>(ids + (long)(start + n) * s_n + c4);
    }
}

template <int D, typename T>
__device__ __forceinline__ void put_ids(half* dst, const typename Vec4<T>::type (&r)[kPer<D>()], int nlen,
                                        const float* mid, const float* step, float c) {
    constexpr int LDK = Smem<D>::LDK;
    static_assert(NT % (D / 4) == 0, "each thread keeps the same 4 columns on every row");
    const int c4 = (threadIdx.x % (D / 4)) * 4;          // the same for every i: load the column constants once
    const float4 md = *reinterpret_cast<const float4*>(mid + c4);
    const float4 sp = *reinterpret_cast<const float4*>(step + c4);
#pragma unroll
    for (int i = 0; i < kPer<D>(); ++i) {
        int idx = threadIdx.x + i * NT;
        int n = idx / (D / 4);
        half2* out = reinterpret_cast<half2*>(dst + n * LDK + c4);
        if (n < nlen) {
            auto v = r[i];
            out[0] = __floats2half2_rn(md.x + sp.x * ((float)v.x - c), md.y + sp.y * ((float)v.y - c));
            out[1] = __floats2half2_rn(md.z + sp.z * ((float)v.z - c), md.w + sp.w * ((float)v.w - c));
        } else {
            out[0] = out[1] = __floats2half2_rn(0.f, 0.f);
        }
    }
}

// Warp w owns output rows [16*(w/2), +16) and column fragments (w%2)*NFC .. +NFC,
// kept in registers for the whole key loop. The online-softmax rescale goes
// through a per-warp 16x16 scratch, and only when one of the warp's rows
// actually raised its running max on this tile (rare after the first tiles).
template <int D, typename T>
__global__ void __launch_bounds__(NT) qvattn_fwd(
    const half* __restrict__ Q, long sq_r, long sq_h,
    const T* __restrict__ KI, const T* __restrict__ VI, long s_n, long s_h,
    const float* __restrict__ KMID, const float* __restrict__ KSTEP,
    const float* __restrict__ VMID, const float* __restrict__ VSTEP,
    const float* __restrict__ KC, const float* __restrict__ VC,
    const int* __restrict__ TS, const int* __restrict__ TL, const int* __restrict__ TSEG,
    float* __restrict__ O, float* __restrict__ M, float* __restrict__ L,
    int q_pos0, int n_tiles, int tps, float scale, int n_rows, int G, int NHKV) {
    using S = Smem<D>;
    constexpr int NFC = D / 16 / 2;                  // column fragments per warp
    extern __shared__ __align__(128) unsigned char sm[];
    half* Qs = reinterpret_cast<half*>(sm + S::q);
    half* KVs = reinterpret_cast<half*>(sm + S::kv);
    float* Ss = reinterpret_cast<float*>(sm + S::s);
    half* Ps = reinterpret_cast<half*>(sm + S::p);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    float* scr = reinterpret_cast<float*>(sm + S::scr) + warp * 256;
    float* m_s = reinterpret_cast<float*>(sm + S::st);
    float* l_s = m_s + BM;
    float* a_s = l_s + BM;

    const int h = blockIdx.x, sp = blockIdx.y, rb = blockIdx.z;
    const int row0 = rb * BM;
    const int fr = warp / 2, fc0 = (warp % 2) * NFC;

    for (int idx = tid; idx < BM * D; idx += NT) {
        int r = idx / D, c = idx % D, row = row0 + r;
        half v = __float2half(0.f);
        if (row < n_rows) v = Q[(long)(row / G) * sq_r + (long)(h * G + row % G) * sq_h + c];
        Qs[r * S::LDQ + c] = v;
    }
    if (tid < BM) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NFC];
#pragma unroll
    for (int f = 0; f < NFC; ++f) wmma::fill_fragment(acc[f], 0.f);
    const int qmax = q_pos0 + (min(row0 + BM, n_rows) - 1) / G;
    __syncthreads();

    const T* ki = KI + (long)h * s_h;
    const T* vi = VI + (long)h * s_h;
    const int t0 = sp * tps, t1 = min(t0 + tps, n_tiles);
    using V4 = typename Vec4<T>::type;
    V4 rk[kPer<D>()], rv[kPer<D>()];
    if (t0 < t1 && TS[t0] <= qmax) {
        fetch_ids<D, T>(rk, ki, s_n, TS[t0], TL[t0]);
        fetch_ids<D, T>(rv, vi, s_n, TS[t0], TL[t0]);
    }
    for (int t = t0; t < t1; ++t) {
        const int start = TS[t];
        if (start > qmax) break;                      // tiles ascend: rest are all masked
        const int nlen = TL[t], seg = TSEG[t];
        const long pb = ((long)seg * NHKV + h) * D;
        put_ids<D, T>(KVs, rk, nlen, KMID + pb, KSTEP + pb, KC[seg * NHKV + h]);
        const bool more = t + 1 < t1 && TS[t + 1] <= qmax;
        if (more) fetch_ids<D, T>(rk, ki, s_n, TS[t + 1], TL[t + 1]);
        __syncthreads();
        {   // S = Q K^T : (BM/16) x (BN/16) = 4 x 2 fragments, one per warp
            const int sr = warp / 2, sc = warp % 2;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sa;
            wmma::fill_fragment(sa, 0.f);
#pragma unroll 4
            for (int k = 0; k < D; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                wmma::load_matrix_sync(a, Qs + sr * 16 * S::LDQ + k, S::LDQ);
                wmma::load_matrix_sync(b, KVs + sc * 16 * S::LDK + k, S::LDK);
                wmma::mma_sync(sa, a, b, sa);
            }
            wmma::store_matrix_sync(Ss + sr * 16 * S::LDS + sc * 16, sa, S::LDS, wmma::mem_row_major);
        }
        __syncthreads();
        {   // online softmax: 4 lanes per row, 8 columns each (BM*4 = NT)
            const int r = tid / 4, j = tid % 4, row = row0 + r;
            const bool row_ok = row < n_rows;
            const int qpos = q_pos0 + row / G;
            float sv[8], mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int n = j * 8 + i;
                bool ok = row_ok && n < nlen && start + n <= qpos;
                sv[i] = ok ? Ss[r * S::LDS + n] * scale : -INFINITY;
                mx = fmaxf(mx, sv[i]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            const float m_old = m_s[r];
            const float m_new = fmaxf(m_old, mx);
            const float m_safe = m_new == -INFINITY ? 0.f : m_new;
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                half p = __float2half(sv[i] == -INFINITY ? 0.f : __expf(sv[i] - m_safe));
                Ps[r * S::LDP + j * 8 + i] = p;
                sum += __half2float(p);               // same rounded p as the P.V product
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            __syncwarp();
            if (j == 0) {
                const float alpha = m_old == -INFINITY ? 0.f : (m_new == m_old ? 1.f : __expf(m_old - m_safe));
                l_s[r] = l_s[r] * alpha + sum;
                m_s[r] = m_new;
                a_s[r] = alpha;
            }
        }
        __syncthreads();
        put_ids<D, T>(KVs, rv, nlen, VMID + pb, VSTEP + pb, VC[seg * NHKV + h]);
        if (more) fetch_ids<D, T>(rv, vi, s_n, TS[t + 1], TL[t + 1]);
        {   // rescale this warp's accumulators only if one of its rows needs it
            const bool need = __any_sync(0xffffffff, lane < 16 && a_s[fr * 16 + lane] != 1.f);
            if (need) {
#pragma unroll
                for (int f = 0; f < NFC; ++f) {
                    wmma::store_matrix_sync(scr, acc[f], 16, wmma::mem_row_major);
                    __syncwarp();
                    for (int e = lane; e < 256; e += 32) scr[e] *= a_s[fr * 16 + e / 16];
                    __syncwarp();
                    wmma::load_matrix_sync(acc[f], scr, 16, wmma::mem_row_major);
                    __syncwarp();
                }
            }
        }
        __syncthreads();
        {   // O += P V
#pragma unroll
            for (int k = 0; k < BN; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::load_matrix_sync(a, Ps + fr * 16 * S::LDP + k, S::LDP);
#pragma unroll
                for (int f = 0; f < NFC; ++f) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
                    wmma::load_matrix_sync(b, KVs + k * S::LDK + (fc0 + f) * 16, S::LDK);
                    wmma::mma_sync(acc[f], a, b, acc[f]);
                }
            }
        }
        __syncthreads();
    }

    const long base = ((long)sp * NHKV + h) * n_rows;
#pragma unroll
    for (int f = 0; f < NFC; ++f) {
        wmma::store_matrix_sync(scr, acc[f], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            int row = row0 + fr * 16 + e / 16;
            if (row < n_rows) O[(base + row) * D + (fc0 + f) * 16 + e % 16] = scr[e];
        }
        __syncwarp();
    }
    if (tid < BM && row0 + tid < n_rows) {
        M[base + row0 + tid] = m_s[tid];
        L[base + row0 + tid] = l_s[tid];
    }
}

template <int D, typename T>
static void launch(torch::Tensor q, torch::Tensor ki, torch::Tensor vi,
                   torch::Tensor kmid, torch::Tensor kstep, torch::Tensor vmid, torch::Tensor vstep,
                   torch::Tensor kc, torch::Tensor vc, torch::Tensor ts, torch::Tensor tl,
                   torch::Tensor tseg, torch::Tensor o, torch::Tensor m, torch::Tensor l,
                   int q_pos0, int tps, double scale, int G, int n_splits) {
    const int nhkv = ki.size(1), n_rows = o.size(2), n_tiles = ts.size(0);
    const size_t smem = Smem<D>::bytes;
    static bool attr[64] = {};                  // the attribute is per device
    const int dev = q.get_device();
    if (!attr[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(qvattn_fwd<D, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr[dev] = true;
    }
    dim3 grid(nhkv, n_splits, (n_rows + BM - 1) / BM);
    qvattn_fwd<D, T><<<grid, NT, smem, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()), q.stride(0), q.stride(1),
        reinterpret_cast<const T*>(ki.data_ptr()), reinterpret_cast<const T*>(vi.data_ptr()), ki.stride(0), ki.stride(1),
        kmid.data_ptr<float>(), kstep.data_ptr<float>(), vmid.data_ptr<float>(), vstep.data_ptr<float>(),
        kc.data_ptr<float>(), vc.data_ptr<float>(),
        ts.data_ptr<int>(), tl.data_ptr<int>(), tseg.data_ptr<int>(),
        o.data_ptr<float>(), m.data_ptr<float>(), l.data_ptr<float>(),
        q_pos0, n_tiles, tps, (float)scale, n_rows, G, nhkv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// q (MQ, Hq, D) fp16, contiguous in D; ki/vi (cap, Hkv, D) uint8; params (nseg, Hkv, D) fp32
// contiguous; kc/vc (nseg, Hkv) fp32; o (S, Hkv, R, D), m/l (S, Hkv, R) fp32.
void fwd(torch::Tensor q, torch::Tensor ki, torch::Tensor vi,
         torch::Tensor kmid, torch::Tensor kstep, torch::Tensor vmid, torch::Tensor vstep,
         torch::Tensor kc, torch::Tensor vc, torch::Tensor ts, torch::Tensor tl, torch::Tensor tseg,
         torch::Tensor o, torch::Tensor m, torch::Tensor l,
         int64_t q_pos0, int64_t tps, double scale, int64_t G) {
    const c10::cuda::OptionalCUDAGuard _g(device_of(q));
    TORCH_CHECK(q.scalar_type() == at::kHalf && q.stride(2) == 1, "q must be fp16, contiguous in D");
    // ids: uint8, or int16 holding uint16 bit patterns (heads that needed 9..16 bits)
    const bool wide = ki.scalar_type() == at::kShort;
    TORCH_CHECK((wide || ki.scalar_type() == at::kByte) && vi.scalar_type() == ki.scalar_type()
                && ki.stride(2) == 1 && ki.strides() == vi.strides(), "ids layout");
    TORCH_CHECK(kmid.is_contiguous() && kstep.is_contiguous() && vmid.is_contiguous() && vstep.is_contiguous()
                && kc.is_contiguous() && vc.is_contiguous(), "params must be contiguous");
    const int d = q.size(2), n_splits = o.size(0);
#define QV_ARGS q, ki, vi, kmid, kstep, vmid, vstep, kc, vc, ts, tl, tseg, o, m, l, q_pos0, tps, scale, G, n_splits
    if (d == 256 && !wide) launch<256, uint8_t>(QV_ARGS);
    else if (d == 256) launch<256, uint16_t>(QV_ARGS);
    else if (d == 128 && !wide) launch<128, uint8_t>(QV_ARGS);
    else if (d == 128) launch<128, uint16_t>(QV_ARGS);
    else TORCH_CHECK(false, "head_dim must be 128 or 256");
}


// ===========================================================================
// Static decode (engine/static_decode.StaticQKV, CUDA graphs) -- every launch shape fixed, lengths read on the device.
//   ids  ki / vi (S, Hkv, cap, D) uint8, params (S, nb, Hkv, D) fp32 = min / step of each QBLK-key block
//   tail tk / tv (S, Hkv, Bt, D) fp16: the open block (+ the keys of a speculative step not yet committed)
//   lc (S) int64 keys held as ids (all visible to every query), vis (S, T) int64 tail keys seen by query t
// grid (S*Hkv, NSP + 1, row blocks): split sp < NSP reads ids tiles [sp*tps*BN, ..) below lc, split NSP the tail.
// qvstatic_combine then merges the splits (log-sum-exp) into o (S, Hq, T, D) fp32 + lse (S, Hq, T): two launches
// per layer (the first version merged with ~30 torch ops: 2/3 of the time on V100, 2026-11-24).
// ===========================================================================
template <int D>
__device__ __forceinline__ void load_dense(half* dst, const half* src, int start, int nlen) {
    constexpr int LDK = Smem<D>::LDK;
    for (int idx = threadIdx.x; idx < BN * D / 8; idx += NT) {
        int n = idx / (D / 8), c8 = (idx % (D / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (n < nlen) v = *reinterpret_cast<const uint4*>(src + (long)(start + n) * D + c8);
        *reinterpret_cast<uint4*>(dst + n * LDK + c8) = v;
    }
}

// one key tile already in KVs (synced): S = Q K^T, online softmax over the visible keys, then V (loadV) and O += P V
template <int D, typename VisF, typename LoadV>
__device__ __forceinline__ void tile_body(const half* Qs, half* KVs, float* Ss, half* Ps, float* scr, float* m_s,
                                          float* l_s, float* a_s,
                                          wmma::fragment<wmma::accumulator, 16, 16, 16, float>* acc,
                                          float scale, VisF vis, LoadV loadV) {
    using S = Smem<D>;
    constexpr int NFC = D / 16 / 2;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, fr = warp / 2, fc0 = (warp % 2) * NFC;
    {
        const int sr = warp / 2, sc = warp % 2;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sa;
        wmma::fill_fragment(sa, 0.f);
#pragma unroll 4
        for (int k = 0; k < D; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::load_matrix_sync(a, Qs + sr * 16 * S::LDQ + k, S::LDQ);
            wmma::load_matrix_sync(b, KVs + sc * 16 * S::LDK + k, S::LDK);
            wmma::mma_sync(sa, a, b, sa);
        }
        wmma::store_matrix_sync(Ss + sr * 16 * S::LDS + sc * 16, sa, S::LDS, wmma::mem_row_major);
    }
    __syncthreads();
    {
        const int r = tid / 4, j = tid % 4;
        float sv[8], mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int n = j * 8 + i;
            sv[i] = vis(r, n) ? Ss[r * S::LDS + n] * scale : -INFINITY;
            mx = fmaxf(mx, sv[i]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        const float m_old = m_s[r];
        const float m_new = fmaxf(m_old, mx);
        const float m_safe = m_new == -INFINITY ? 0.f : m_new;
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half p = __float2half(sv[i] == -INFINITY ? 0.f : __expf(sv[i] - m_safe));
            Ps[r * S::LDP + j * 8 + i] = p;
            sum += __half2float(p);
        }
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        sum += __shfl_xor_sync(0xffffffff, sum, 2);
        __syncwarp();
        if (j == 0) {
            const float alpha = m_old == -INFINITY ? 0.f : (m_new == m_old ? 1.f : __expf(m_old - m_safe));
            l_s[r] = l_s[r] * alpha + sum;
            m_s[r] = m_new;
            a_s[r] = alpha;
        }
    }
    __syncthreads();
    loadV();
    {
        const bool need = __any_sync(0xffffffff, lane < 16 && a_s[fr * 16 + lane] != 1.f);
        if (need) {
#pragma unroll
            for (int f = 0; f < NFC; ++f) {
                wmma::store_matrix_sync(scr, acc[f], 16, wmma::mem_row_major);
                __syncwarp();
                for (int e = lane; e < 256; e += 32) scr[e] *= a_s[fr * 16 + e / 16];
                __syncwarp();
                wmma::load_matrix_sync(acc[f], scr, 16, wmma::mem_row_major);
                __syncwarp();
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BN; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::load_matrix_sync(a, Ps + fr * 16 * S::LDP + k, S::LDP);
#pragma unroll
        for (int f = 0; f < NFC; ++f) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
            wmma::load_matrix_sync(b, KVs + k * S::LDK + (fc0 + f) * 16, S::LDK);
            wmma::mma_sync(acc[f], a, b, acc[f]);
        }
    }
    __syncthreads();
}

template <int D>
__global__ void __launch_bounds__(NT) qvattn_static(
    const half* __restrict__ Q, long sq_s, long sq_h, long sq_t,
    const uint8_t* __restrict__ KI, const uint8_t* __restrict__ VI, long si_s, long si_h,
    const float* __restrict__ KMIN, const float* __restrict__ KSTEP,
    const float* __restrict__ VMIN, const float* __restrict__ VSTEP, long p_s,
    const half* __restrict__ TK, const half* __restrict__ TV, long st_s, long st_h, int Bt,
    const long* __restrict__ LC, const long* __restrict__ VIS, int T,
    float* __restrict__ O, float* __restrict__ M, float* __restrict__ L,
    int tps, int QBLK, float scale, int n_rows, int G, int NHKV) {
    using S = Smem<D>;
    constexpr int NFC = D / 16 / 2;
    extern __shared__ __align__(128) unsigned char sm[];
    __shared__ int vrow[BM];
    half* Qs = reinterpret_cast<half*>(sm + S::q);
    half* KVs = reinterpret_cast<half*>(sm + S::kv);
    float* Ss = reinterpret_cast<float*>(sm + S::s);
    half* Ps = reinterpret_cast<half*>(sm + S::p);
    const int tid = threadIdx.x, warp = tid / 32, fr = warp / 2, fc0 = (warp % 2) * NFC;
    float* scr = reinterpret_cast<float*>(sm + S::scr) + warp * 256;
    float* m_s = reinterpret_cast<float*>(sm + S::st);
    float* l_s = m_s + BM;
    float* a_s = l_s + BM;
    const int slot = blockIdx.x / NHKV, h = blockIdx.x % NHKV, sp = blockIdx.y, rb = blockIdx.z;
    const int NSP = gridDim.y - 1, row0 = rb * BM;

    for (int idx = tid; idx < BM * D; idx += NT) {             // rows r = t * G + g
        int r = idx / D, c = idx % D, row = row0 + r;
        half v = __float2half(0.f);
        if (row < n_rows) v = Q[slot * sq_s + (long)(h * G + row % G) * sq_h + (long)(row / G) * sq_t + c];
        Qs[r * S::LDQ + c] = v;
    }
    if (tid < BM) {
        m_s[tid] = -INFINITY; l_s[tid] = 0.f;
        int row = row0 + tid;
        vrow[tid] = row < n_rows ? (int)VIS[slot * T + row / G] : 0;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NFC];
#pragma unroll
    for (int f = 0; f < NFC; ++f) wmma::fill_fragment(acc[f], 0.f);
    __syncthreads();

    if (sp < NSP) {                                            // closed blocks: 8-bit ids, every key below lc visible
        const int lc = (int)LC[slot];
        const uint8_t* ki = KI + slot * si_s + h * si_h;
        const uint8_t* vi = VI + slot * si_s + h * si_h;
        const int a = sp * tps * BN, b = min(a + tps * BN, lc);
        for (int start = a; start < b; start += BN) {
            const int nlen = min(BN, b - start);
            const long pb = slot * p_s + ((long)(start / QBLK) * NHKV + h) * D;
            load_tile<D, uint8_t>(KVs, ki, D, start, nlen, KMIN + pb, KSTEP + pb, 0.f);
            __syncthreads();
            tile_body<D>(Qs, KVs, Ss, Ps, scr, m_s, l_s, a_s, acc, scale,
                         [&](int r, int n) { return row0 + r < n_rows && n < nlen; },
                         [&]() { load_tile<D, uint8_t>(KVs, vi, D, start, nlen, VMIN + pb, VSTEP + pb, 0.f); });
        }
    } else {                                                   // the dense tail: causal per query
        const int last = min(row0 + BM, n_rows) - 1 - row0;
        const int vmax = min(vrow[last], Bt);
        const half* tk = TK + slot * st_s + h * st_h;
        const half* tv = TV + slot * st_s + h * st_h;
        for (int start = 0; start < vmax; start += BN) {
            const int nlen = min(BN, Bt - start);
            load_dense<D>(KVs, tk, start, nlen);
            __syncthreads();
            tile_body<D>(Qs, KVs, Ss, Ps, scr, m_s, l_s, a_s, acc, scale,
                         [&](int r, int n) { return row0 + r < n_rows && n < nlen && start + n < vrow[r]; },
                         [&]() { load_dense<D>(KVs, tv, start, nlen); });
        }
    }
    const long base = (((long)slot * gridDim.y + sp) * NHKV + h) * n_rows;
#pragma unroll
    for (int f = 0; f < NFC; ++f) {
        wmma::store_matrix_sync(scr, acc[f], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = threadIdx.x % 32; e < 256; e += 32) {
            int row = row0 + fr * 16 + e / 16;
            if (row < n_rows) O[(base + row) * D + (fc0 + f) * 16 + e % 16] = scr[e];
        }
        __syncwarp();
    }
    if (tid < BM && row0 + tid < n_rows) {
        M[base + row0 + tid] = m_s[tid];
        L[base + row0 + tid] = l_s[tid];
    }
}

// the splits of one (slot, kv head, row) -> out (S, Hq, T, D) fp32 normalised, lse (S, Hq, T)
__global__ void qvstatic_combine(const float* __restrict__ O, const float* __restrict__ M, const float* __restrict__ L,
                                 float* __restrict__ OUT, float* __restrict__ LSE, int NSP1, int NHKV, int n_rows,
                                 int D, int G, int T) {
    const int idx = blockIdx.x, r = idx % n_rows, h = (idx / n_rows) % NHKV, s = idx / (n_rows * NHKV);
    float mx = -INFINITY;
    for (int p = 0; p < NSP1; ++p) {
        long i = (((long)s * NSP1 + p) * NHKV + h) * n_rows + r;
        if (L[i] > 0.f) mx = fmaxf(mx, M[i]);
    }
    const int t = r / G, g = r % G, Hq = NHKV * G;
    const long oo = (((long)s * Hq + h * G + g) * T + t);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float den = 0.f, acc = 0.f;
        for (int p = 0; p < NSP1; ++p) {
            long i = (((long)s * NSP1 + p) * NHKV + h) * n_rows + r;
            float l = L[i];
            if (l > 0.f) { float w = __expf(M[i] - mx); den += w * l; acc += w * O[i * D + d]; }
        }
        OUT[oo * D + d] = den > 0.f ? acc / den : 0.f;
        if (d == 0) LSE[oo] = den > 0.f ? mx + logf(den) : -INFINITY;
    }
}

template <int D>
static void launch_static(torch::Tensor q, torch::Tensor ki, torch::Tensor vi, torch::Tensor kmin,
                          torch::Tensor kstep, torch::Tensor vmin, torch::Tensor vstep, torch::Tensor tk,
                          torch::Tensor tv, torch::Tensor lc, torch::Tensor vis, torch::Tensor o, torch::Tensor m,
                          torch::Tensor l, torch::Tensor out, torch::Tensor lse, int tps, int qblk, double scale) {
    const int S = q.size(0), Hq = q.size(1), T = q.size(2), nhkv = ki.size(1), G = Hq / nhkv, n_rows = T * G;
    const int nsp1 = o.size(1);
    const size_t smem = Smem<D>::bytes;
    static bool attr[64] = {};
    const int dev = q.get_device();
    if (!attr[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(qvattn_static<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr[dev] = true;
    }
    auto st = c10::cuda::getCurrentCUDAStream();
    dim3 grid(S * nhkv, nsp1, (n_rows + BM - 1) / BM);
    qvattn_static<D><<<grid, NT, smem, st>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()), q.stride(0), q.stride(1), q.stride(2),
        ki.data_ptr<uint8_t>(), vi.data_ptr<uint8_t>(), ki.stride(0), ki.stride(1),
        kmin.data_ptr<float>(), kstep.data_ptr<float>(), vmin.data_ptr<float>(), vstep.data_ptr<float>(), kmin.stride(0),
        reinterpret_cast<const half*>(tk.data_ptr<at::Half>()), reinterpret_cast<const half*>(tv.data_ptr<at::Half>()),
        tk.stride(0), tk.stride(1), (int)tk.size(2),
        lc.data_ptr<int64_t>(), vis.data_ptr<int64_t>(), T,
        o.data_ptr<float>(), m.data_ptr<float>(), l.data_ptr<float>(), tps, qblk, (float)scale, n_rows, G, nhkv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    qvstatic_combine<<<S * nhkv * n_rows, D, 0, st>>>(o.data_ptr<float>(), m.data_ptr<float>(), l.data_ptr<float>(),
        out.data_ptr<float>(), lse.data_ptr<float>(), nsp1, nhkv, n_rows, D, G, T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// q (S, Hq, T, D) fp16 (D contiguous); o (S, NSP+1, Hkv, T*G, D), m / l (S, NSP+1, Hkv, T*G) fp32 scratch;
// out (S, Hq, T, D) fp32, lse (S, Hq, T) fp32 contiguous
void fwd_static(torch::Tensor q, torch::Tensor ki, torch::Tensor vi, torch::Tensor kmin, torch::Tensor kstep,
                torch::Tensor vmin, torch::Tensor vstep, torch::Tensor tk, torch::Tensor tv, torch::Tensor lc,
                torch::Tensor vis, torch::Tensor o, torch::Tensor m, torch::Tensor l, torch::Tensor out,
                torch::Tensor lse, int64_t tps, int64_t qblk, double scale) {
    const c10::cuda::OptionalCUDAGuard _g(device_of(q));
    TORCH_CHECK(q.scalar_type() == at::kHalf && q.stride(3) == 1, "q must be fp16, contiguous in D");
    TORCH_CHECK(ki.scalar_type() == at::kByte && ki.stride(3) == 1 && ki.stride(2) == ki.size(3)
                && ki.strides() == vi.strides(), "ids (S, Hkv, cap, D)");
    TORCH_CHECK(kmin.stride(3) == 1 && kmin.stride(2) == kmin.size(3) && kmin.stride(1) == kmin.size(2) * kmin.size(3)
                && kmin.strides() == kstep.strides() && kmin.strides() == vmin.strides()
                && kmin.strides() == vstep.strides(), "params (S, nb, Hkv, D)");
    TORCH_CHECK(tk.scalar_type() == at::kHalf && tk.stride(3) == 1 && tk.stride(2) == tk.size(3)
                && tk.strides() == tv.strides(), "tail (S, Hkv, Bt, D) fp16");
    TORCH_CHECK(lc.scalar_type() == at::kLong && vis.scalar_type() == at::kLong && vis.is_contiguous(), "lc / vis");
    TORCH_CHECK(o.is_contiguous() && m.is_contiguous() && l.is_contiguous() && out.is_contiguous()
                && lse.is_contiguous(), "outputs");
    const int d = q.size(3);
    if (d == 256) launch_static<256>(q, ki, vi, kmin, kstep, vmin, vstep, tk, tv, lc, vis, o, m, l, out, lse, tps, qblk, scale);
    else if (d == 128) launch_static<128>(q, ki, vi, kmin, kstep, vmin, vstep, tk, tv, lc, vis, o, m, l, out, lse, tps, qblk, scale);
    else TORCH_CHECK(false, "head_dim must be 128 or 256");
}


// ===========================================================================
// Gated DeltaNet chunked forward (fla chunk_gated_delta_rule_fwd_h +
// chunk_fwd_o) with WMMA for sm_70, whose Triton has no tensor-core dot.
// Same math and the same stored dtypes as fla (h and v_new rounded to the
// model dtype; A rounded to the model dtype before A @ v); g is the
// chunk-local cumsum in log2 units (fla scales by 1/ln 2). K = V = 128,
// BT = 64, no gk / g_gamma / varlen / state_v_first -- the wrapper falls
// back to fla for anything else.
// ===========================================================================
#include <cuda_bf16.h>

template <typename T> __device__ __forceinline__ float tof(T x);
template <> __device__ __forceinline__ float tof<at::Half>(at::Half x) { return __half2float(*reinterpret_cast<half*>(&x)); }
template <> __device__ __forceinline__ float tof<at::BFloat16>(at::BFloat16 x) { return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&x)); }
template <typename T> __device__ __forceinline__ T fromf(float x);
template <> __device__ __forceinline__ at::Half fromf<at::Half>(float x) { half h = __float2half_rn(x); return *reinterpret_cast<at::Half*>(&h); }
template <> __device__ __forceinline__ at::BFloat16 fromf<at::BFloat16>(float x) { __nv_bfloat16 h = __float2bfloat16_rn(x); return *reinterpret_cast<at::BFloat16*>(&h); }
template <typename T> __device__ __forceinline__ float rnd(float x) { return tof<T>(fromf<T>(x)); }

namespace gdn {
constexpr int BT = 64, KD = 128, VD = 128, BVH = 32, NTH = 128;
constexpr int LD136 = 136, LD36 = 36, LD40 = 40, LD68 = 68, LD72 = 72, LD132 = 132;
}

// grid (VD/BVH, B*HV). State slice (KD x BVH) lives in WMMA accumulators.
template <typename T>
__global__ void __launch_bounds__(gdn::NTH) gdn_fwd_h(
    const T* __restrict__ k, const T* __restrict__ w, const T* __restrict__ u,
    const float* __restrict__ g, T* __restrict__ hout, T* __restrict__ vnew,
    const float* __restrict__ h0, float* __restrict__ ht, int Tn, int H, int HV, int NT) {
    using namespace gdn;
    extern __shared__ __align__(128) unsigned char sm[];
    half* Ws = reinterpret_cast<half*>(sm);
    half* Ks = Ws + BT * LD136;
    float* S32 = reinterpret_cast<float*>(Ks + BT * LD136);
    half* S16 = reinterpret_cast<half*>(S32 + KD * LD36);
    float* V32 = reinterpret_cast<float*>(S16 + KD * LD40);
    half* V16 = reinterpret_cast<half*>(V32 + BT * LD36);

    const int iv = blockIdx.x, bh = blockIdx.y, b = bh / HV, hv = bh % HV, hk = hv / (HV / H);
    const int tid = threadIdx.x, warp = tid / 32;
    const int v0 = iv * BVH;
    const long tok0 = (long)b * Tn;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> st[4];   // rows (2w, 2w+1) x cols (0,1)
    if (h0) {
        for (int e = tid; e < KD * BVH; e += NTH) {
            int kk = e / BVH, j = e % BVH;
            S32[kk * LD36 + j] = h0[((long)bh * KD + kk) * VD + v0 + j];
        }
        __syncthreads();
        for (int f = 0; f < 4; ++f)
            wmma::load_matrix_sync(st[f], S32 + ((2 * warp + f / 2) * 16) * LD36 + (f % 2) * 16, LD36, wmma::mem_row_major);
    } else {
        for (int f = 0; f < 4; ++f) wmma::fill_fragment(st[f], 0.f);
    }

    for (int it = 0; it < NT; ++it) {
        const int t0 = it * BT;
        const int nv = min(BT, Tn - t0);
        __syncthreads();
        for (int f = 0; f < 4; ++f)
            wmma::store_matrix_sync(S32 + ((2 * warp + f / 2) * 16) * LD36 + (f % 2) * 16, st[f], LD36, wmma::mem_row_major);
        // w, k chunk -> fp16 smem (zero past the end)
        for (int e = tid; e < BT * KD; e += NTH) {
            int t = e / KD, c = e % KD;
            float wv = 0.f, kv = 0.f;
            if (t < nv) {
                wv = tof<T>(w[((tok0 + t0 + t) * HV + hv) * KD + c]);
                kv = tof<T>(k[((tok0 + t0 + t) * H + hk) * KD + c]);
            }
            Ws[t * LD136 + c] = __float2half(wv);
            Ks[t * LD136 + c] = __float2half(kv);
        }
        __syncthreads();
        // h[it] = state (model dtype); fp16 copy for the w @ h product (same rounding)
        for (int e = tid; e < KD * BVH; e += NTH) {
            int kk = e / BVH, j = e % BVH;
            T hv_ = fromf<T>(S32[kk * LD36 + j]);
            hout[((((long)b * NT + it) * HV + hv) * KD + kk) * VD + v0 + j] = hv_;
            S16[kk * LD40 + j] = __float2half(tof<T>(hv_));
        }
        __syncthreads();
        {   // V32 = W (BT x KD) @ S16 (KD x BVH): 4 x 2 frags, 2 per warp
            for (int f = 0; f < 2; ++f) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.f);
                for (int kk = 0; kk < KD; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bb;
                    wmma::load_matrix_sync(a, Ws + warp * 16 * LD136 + kk, LD136);
                    wmma::load_matrix_sync(bb, S16 + kk * LD40 + f * 16, LD40);
                    wmma::mma_sync(acc, a, bb, acc);
                }
                wmma::store_matrix_sync(V32 + warp * 16 * LD36 + f * 16, acc, LD36, wmma::mem_row_major);
            }
        }
        __syncthreads();
        const float glast = g[(tok0 + t0 + nv - 1) * HV + hv];
        for (int e = tid; e < BT * BVH; e += NTH) {
            int t = e / BVH, j = e % BVH;
            float sc = 0.f;
            if (t < nv) {
                long idx = ((tok0 + t0 + t) * HV + hv) * VD + v0 + j;
                T vn = fromf<T>(tof<T>(u[idx]) - V32[t * LD36 + j]);
                vnew[idx] = vn;
                sc = rnd<T>(tof<T>(vn) * exp2f(glast - g[(tok0 + t0 + t) * HV + hv]));
            }
            V16[t * LD40 + j] = __float2half(sc);
        }
        __syncthreads();
        {   // state = state * 2^glast + K^T (KD x BT) @ V16 (BT x BVH)
            const float dec = exp2f(glast);
            for (int f = 0; f < 4; ++f)
                for (int i = 0; i < st[f].num_elements; ++i) st[f].x[i] *= dec;
            for (int t = 0; t < BT; t += 16) {
                for (int f = 0; f < 4; ++f) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bb;
                    wmma::load_matrix_sync(a, Ks + t * LD136 + (2 * warp + f / 2) * 16, LD136);
                    wmma::load_matrix_sync(bb, V16 + t * LD40 + (f % 2) * 16, LD40);
                    wmma::mma_sync(st[f], a, bb, st[f]);
                }
            }
        }
    }
    if (ht) {
        __syncthreads();
        for (int f = 0; f < 4; ++f)
            wmma::store_matrix_sync(S32 + ((2 * warp + f / 2) * 16) * LD36 + (f % 2) * 16, st[f], LD36, wmma::mem_row_major);
        __syncthreads();
        for (int e = tid; e < KD * BVH; e += NTH) {
            int kk = e / BVH, j = e % BVH;
            ht[((long)bh * KD + kk) * VD + v0 + j] = S32[kk * LD36 + j];
        }
    }
}

// grid (NT, B*HV): one 64-token chunk x one value head, all VD columns.
template <typename T>
__global__ void __launch_bounds__(gdn::NTH) gdn_fwd_o(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    const T* __restrict__ h, const float* __restrict__ g, T* __restrict__ o,
    float scale, int Tn, int H, int HV, int NT) {
    using namespace gdn;
    extern __shared__ __align__(128) unsigned char sm[];
    half* R1 = reinterpret_cast<half*>(sm);          // q raw, later v
    half* R2 = R1 + BT * LD136;                      // q * 2^g
    half* R3 = R2 + BT * LD136;                      // k, later P
    unsigned char* R4 = reinterpret_cast<unsigned char*>(R3 + BT * LD136);   // A fp32 / h fp16 / O fp32
    float* gs = reinterpret_cast<float*>(R4 + sizeof(half) * KD * LD136);

    const int it = blockIdx.x, bh = blockIdx.y, b = bh / HV, hv = bh % HV, hk = hv / (HV / H);
    const int tid = threadIdx.x, warp = tid / 32;
    const int t0 = it * BT, nv = min(BT, Tn - t0);
    const long tok0 = (long)b * Tn + t0;

    if (tid < BT) gs[tid] = tid < nv ? g[(tok0 + tid) * HV + hv] : 0.f;
    __syncthreads();
    for (int e = tid; e < BT * KD; e += NTH) {
        int t = e / KD, c = e % KD;
        float qv = 0.f, kv = 0.f;
        if (t < nv) {
            qv = tof<T>(q[((tok0 + t) * H + hk) * KD + c]);
            kv = tof<T>(k[((tok0 + t) * H + hk) * KD + c]);
        }
        R1[t * LD136 + c] = __float2half(qv);
        R2[t * LD136 + c] = __float2half(qv * exp2f(gs[t]));
        R3[t * LD136 + c] = __float2half(kv);
    }
    __syncthreads();
    float* A32 = reinterpret_cast<float*>(R4);
    for (int f = 0; f < 4; ++f) {    // A = q k^T, row frag = warp
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
        for (int kk = 0; kk < KD; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bb;
            wmma::load_matrix_sync(a, R1 + warp * 16 * LD136 + kk, LD136);
            wmma::load_matrix_sync(bb, R3 + f * 16 * LD136 + kk, LD136);
            wmma::mma_sync(acc, a, bb, acc);
        }
        wmma::store_matrix_sync(A32 + warp * 16 * LD68 + f * 16, acc, LD68, wmma::mem_row_major);
    }
    __syncthreads();
    half* P = R3;
    for (int e = tid; e < BT * BT; e += NTH) {
        int t = e / BT, s2 = e % BT;
        float a = (s2 <= t && t < nv) ? A32[t * LD68 + s2] * exp2f(gs[t] - gs[s2]) : 0.f;
        P[t * LD72 + s2] = __float2half(rnd<T>(a));
    }
    __syncthreads();
    half* Hs = reinterpret_cast<half*>(R4);
    const T* hp = h + (((long)b * NT + it) * HV + hv) * KD * VD;
    for (int e = tid; e < KD * VD; e += NTH) {
        int kk = e / VD, j = e % VD;
        Hs[kk * LD136 + j] = __float2half(tof<T>(hp[e]));
    }
    for (int e = tid; e < BT * VD; e += NTH) {
        int t = e / VD, j = e % VD;
        R1[t * LD136 + j] = __float2half(t < nv ? tof<T>(v[((tok0 + t) * HV + hv) * VD + j]) : 0.f);
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    for (int f = 0; f < 8; ++f) wmma::fill_fragment(acc[f], 0.f);
    for (int kk = 0; kk < KD; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::load_matrix_sync(a, R2 + warp * 16 * LD136 + kk, LD136);
        for (int f = 0; f < 8; ++f) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bb;
            wmma::load_matrix_sync(bb, Hs + kk * LD136 + f * 16, LD136);
            wmma::mma_sync(acc[f], a, bb, acc[f]);
        }
    }
    for (int s0 = 0; s0 < BT; s0 += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::load_matrix_sync(a, P + warp * 16 * LD72 + s0, LD72);
        for (int f = 0; f < 8; ++f) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bb;
            wmma::load_matrix_sync(bb, R1 + s0 * LD136 + f * 16, LD136);
            wmma::mma_sync(acc[f], a, bb, acc[f]);
        }
    }
    __syncthreads();
    float* O32 = reinterpret_cast<float*>(R4);
    for (int f = 0; f < 8; ++f)
        wmma::store_matrix_sync(O32 + warp * 16 * LD132 + f * 16, acc[f], LD132, wmma::mem_row_major);
    __syncthreads();
    for (int e = tid; e < BT * VD; e += NTH) {
        int t = e / VD, j = e % VD;
        if (t < nv) o[((tok0 + t) * HV + hv) * VD + j] = fromf<T>(O32[t * LD132 + j] * scale);
    }
}

static size_t gdn_h_smem() {
    using namespace gdn;
    return sizeof(half) * BT * LD136 * 2 + sizeof(float) * KD * LD36 + sizeof(half) * KD * LD40
         + sizeof(float) * BT * LD36 + sizeof(half) * BT * LD40;
}
static size_t gdn_o_smem() {
    using namespace gdn;
    return sizeof(half) * BT * LD136 * 3 + sizeof(half) * KD * LD136 + sizeof(float) * BT;
}

template <typename T>
static void gdn_launch(torch::Tensor k, torch::Tensor w, torch::Tensor u, torch::Tensor g,
                       torch::Tensor hout, torch::Tensor vnew, c10::optional<torch::Tensor> h0,
                       c10::optional<torch::Tensor> ht) {
    using namespace gdn;
    const int B = k.size(0), Tn = k.size(1), H = k.size(2), HV = u.size(2), NT = hout.size(1);
    static bool attr[64] = {};
    const int dev = k.get_device();
    if (!attr[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_fwd_h<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gdn_h_smem()));
        attr[dev] = true;
    }
    gdn_fwd_h<T><<<dim3(VD / BVH, B * HV), NTH, gdn_h_smem(), c10::cuda::getCurrentCUDAStream()>>>(
        k.data_ptr<T>(), w.data_ptr<T>(), u.data_ptr<T>(), g.data_ptr<float>(),
        hout.data_ptr<T>(), vnew.data_ptr<T>(),
        h0.has_value() ? h0->data_ptr<float>() : nullptr,
        ht.has_value() ? ht->data_ptr<float>() : nullptr, Tn, H, HV, NT);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_h(torch::Tensor k, torch::Tensor w, torch::Tensor u, torch::Tensor g,
           torch::Tensor hout, torch::Tensor vnew, c10::optional<torch::Tensor> h0,
           c10::optional<torch::Tensor> ht) {
    const c10::cuda::OptionalCUDAGuard _g(device_of(k));
    TORCH_CHECK(k.is_contiguous() && w.is_contiguous() && u.is_contiguous() && g.is_contiguous()
                && g.scalar_type() == at::kFloat, "gdn_h layout");
    if (k.scalar_type() == at::kHalf) gdn_launch<at::Half>(k, w, u, g, hout, vnew, h0, ht);
    else if (k.scalar_type() == at::kBFloat16) gdn_launch<at::BFloat16>(k, w, u, g, hout, vnew, h0, ht);
    else TORCH_CHECK(false, "gdn_h dtype");
}

template <typename T>
static void gdn_o_launch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor h,
                         torch::Tensor g, torch::Tensor o, double scale) {
    using namespace gdn;
    const int B = q.size(0), Tn = q.size(1), H = q.size(2), HV = v.size(2), NT = h.size(1);
    static bool attr[64] = {};
    const int dev = q.get_device();
    if (!attr[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_fwd_o<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gdn_o_smem()));
        attr[dev] = true;
    }
    gdn_fwd_o<T><<<dim3(NT, B * HV), NTH, gdn_o_smem(), c10::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), h.data_ptr<T>(), g.data_ptr<float>(),
        o.data_ptr<T>(), (float)scale, Tn, H, HV, NT);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_o(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor h, torch::Tensor g,
           torch::Tensor o, double scale) {
    const c10::cuda::OptionalCUDAGuard _g(device_of(q));
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && h.is_contiguous()
                && g.is_contiguous() && g.scalar_type() == at::kFloat, "gdn_o layout");
    if (q.scalar_type() == at::kHalf) gdn_o_launch<at::Half>(q, k, v, h, g, o, scale);
    else if (q.scalar_type() == at::kBFloat16) gdn_o_launch<at::BFloat16>(q, k, v, h, g, o, scale);
    else TORCH_CHECK(false, "gdn_o dtype");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("fwd", &fwd);
    mod.def("fwd_static", &fwd_static);
    mod.def("gdn_h", &gdn_h);
    mod.def("gdn_o", &gdn_o);
}
