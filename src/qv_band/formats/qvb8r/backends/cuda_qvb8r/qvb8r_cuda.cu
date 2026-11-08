// qvb8r CUDA kernels: rANS-coded QVB8 band ids decoded on the fly (format: formats/qvb8r/codec.py, container.py).
//
// Every kernel walks a tile the same way: lane j of a warp owns a 64-bit rANS state j (two registers, xh:xl); per
// decode step each lane looks its state up in the class's 4096-entry slot table (shared memory, one class per
// block), gets symbol s for column 32*step + j and advances x' = f * (x >> 12) + d. Steps go in PAIRS: after two
// symbols, lanes whose state fell below 2^32 take the next 32-bit word of the tile's shared stream in lane order
// (ballot + popc) -- one renormalisation per two symbols (P % 128 == 0, so a pair never straddles a row). Words
// come from a per-warp shared-memory ring refilled from registers loaded half a ring ahead.
// id = s + drow[row] (rows are coded centred on one symbol; an integer add, exact).
// Band value = fmaf(id, tau_s[row], min_s[row]) in fp32 (the WMMA path rounds it to fp16).
//
//   gemv      M = 1        fp32 X, warp per tile
//   multi     M = 2..8     X transposed (P, MB) + vector loads; bf16 X read as raw bits (exact)
//   tc        M = 9..16    16 warps in lock-step, 16 x 128 fp16 weight tile in smem, WMMA m16n16k16 (sm_70+)
//   decode    dense rows   (for M > 16 chunked F.linear, and for verification)
//   fix_escapes            y[m, q] += sum over q's escapes of x[m, c] * delta, in stored order (deterministic)
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
using namespace nvcuda;

#define WPB 8          // tiles (warps) per block: gemv / multi / decode
#define TC_WARPS 16    // tiles (warps) per block: tc
#define TC_SB 4        // tc: decode steps per barrier (even: whole pairs)
#define RING 256       // 32-bit words per warp ring: a pair takes <= 32, refill every RING / 2, so never overrun

#define CHECK_CUDA_INPUTS(...) do { for (const torch::Tensor& _t : {__VA_ARGS__}) \
    TORCH_CHECK(_t.is_cuda() && _t.is_contiguous(), "qvb8r: inputs must be contiguous CUDA tensors"); } while (0)

// ---------------------------------------------------------------------------------------------------------------
// shared tile walker pieces
struct Stream {
    uint32_t xl, xh; int ptr, sb; uint32_t pre[RING / 64]; const uint32_t* wp;   // cursor relative to the tile's words
};
__device__ __forceinline__ void stream_init(Stream& z, uint32_t* rg, const uint32_t* __restrict__ words,
                                            const int64_t* __restrict__ tile_off, const uint2* __restrict__ states,
                                            int tile, int lane) {
    z.wp = words + tile_off[tile];
    const uint2 sv = states[(int64_t)tile * 32 + lane];
    z.xl = sv.x; z.xh = sv.y; z.ptr = z.sb = 0;
    #pragma unroll
    for (int q = 0; q < RING / 32; ++q) rg[32 * q + lane] = __ldg(z.wp + 32 * q + lane);
    #pragma unroll
    for (int q = 0; q < RING / 64; ++q) z.pre[q] = __ldg(z.wp + RING + 32 * q + lane);
}
// one symbol: x' = f * (x >> 12) + d on the 64-bit state
__device__ __forceinline__ uint32_t stream_dec(Stream& z, const uint32_t* st) {
    const uint32_t E = st[z.xl & 4095u];
    const uint32_t f = (E >> 22) + 1u, d = (E >> 11) & 2047u;
    const uint32_t ql = (z.xl >> 12) | (z.xh << 20), qh = z.xh >> 12;
    const uint32_t lo = f * ql, hi = f * qh + __umulhi(f, ql);
    z.xl = lo + d; z.xh = hi + (z.xl < lo ? 1u : 0u);
    return E & 2047u;
}
// two symbols, then one renormalisation
__device__ __forceinline__ void stream_pair(Stream& z, const uint32_t* st, uint32_t* rg, int lane, uint32_t lt,
                                            uint32_t& s1, uint32_t& s2) {
    s1 = stream_dec(z, st); s2 = stream_dec(z, st);
    const bool need = z.xh == 0u;
    const uint32_t mk = __ballot_sync(0xffffffff, need);
    if (need) { z.xh = z.xl; z.xl = rg[(z.ptr + __popc(mk & lt)) & (RING - 1)]; }
    z.ptr += __popc(mk);
    if (z.ptr - z.sb >= RING / 2) {
        #pragma unroll
        for (int q = 0; q < RING / 64; ++q) rg[(z.sb + RING + 32 * q + lane) & (RING - 1)] = z.pre[q];
        __syncwarp();
        z.sb += RING / 2;
        #pragma unroll
        for (int q = 0; q < RING / 64; ++q) z.pre[q] = __ldg(z.wp + z.sb + RING + 32 * q + lane);
    }
}
#define BLOCK_PROLOGUE(NWARPS)                                                                              \
    const int cls = blocks[3 * blockIdx.x], tb = blocks[3 * blockIdx.x + 1], ntb = blocks[3 * blockIdx.x + 2]; \
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) st[i] = tab[(int64_t)cls * 4096 + i];             \
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;                                                \
    const uint32_t lt = (1u << lane) - 1u;                                                                  \
    const int S = P >> 5;

// ---------------------------------------------------------------------------------------------------------------
// M = 1
template <bool DUMP>
__global__ void __launch_bounds__(32 * WPB) qvb8r_gemv(const float* __restrict__ X, const uint32_t* __restrict__ words,
    const int64_t* __restrict__ tile_off, const uint2* __restrict__ states, const int64_t* __restrict__ tiles,
    const int32_t* __restrict__ blocks, const uint32_t* __restrict__ tab, const int32_t* __restrict__ perm,
    const float* __restrict__ mn, const float* __restrict__ ts, const int32_t* __restrict__ drow, float* __restrict__ Y, int16_t* __restrict__ dump, int P) {
    __shared__ uint32_t st[4096];
    __shared__ uint32_t ring[WPB][RING];
    BLOCK_PROLOGUE(WPB)
    __syncthreads();
    if (w >= ntb) return;
    const int tile = tb + w;
    const int r0 = (int)tiles[3 * tile + 1], nr = (int)tiles[3 * tile + 2];
    uint32_t* rg = ring[w];
    Stream z; stream_init(z, rg, words, tile_off, states, tile, lane);
    __syncwarp();
    for (int r = 0; r < nr; ++r) {
        const int orow = perm[r0 + r];
        const float m = mn[orow], t = ts[orow];
        const int dr = drow[orow];
        float acc = 0.f, xc[4], xn[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) xc[j] = __ldg(X + j * 32 + lane);
        for (int s0 = 0; s0 < S; s0 += 4) {                // P % 128 == 0
            #pragma unroll
            for (int j = 0; j < 4; ++j) xn[j] = (s0 + 4 < S) ? __ldg(X + (s0 + 4 + j) * 32 + lane) : 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j += 2) {
                uint32_t s1, s2; stream_pair(z, st, rg, lane, lt, s1, s2);
                const int i1 = (int)s1 + dr, i2 = (int)s2 + dr;
                acc = fmaf(xc[j], fmaf((float)i1, t, m), acc);
                acc = fmaf(xc[j + 1], fmaf((float)i2, t, m), acc);
                if (DUMP) { dump[(int64_t)orow * P + (s0 + j) * 32 + lane] = (int16_t)i1;
                            dump[(int64_t)orow * P + (s0 + j + 1) * 32 + lane] = (int16_t)i2; }
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) xc[j] = xn[j];
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) Y[orow] = acc;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// M = 2..8 (MB = 2, 4, 8): each id decoded once, used for all rows. X is (P, MB): a lane's MB activations of one
// column are contiguous. XT = float, or uint16_t holding bf16 bits (bf16 -> fp32 is a 16-bit shift, exact).
template <int MB>
__device__ __forceinline__ void ldx(float* v, const float* p) {
    if constexpr (MB % 4 == 0) {
        #pragma unroll
        for (int i = 0; i < MB; i += 4) { const float4 q = __ldg(reinterpret_cast<const float4*>(p + i)); v[i] = q.x; v[i + 1] = q.y; v[i + 2] = q.z; v[i + 3] = q.w; }
    } else {
        const float2 q = __ldg(reinterpret_cast<const float2*>(p)); v[0] = q.x; v[1] = q.y;
    }
}
template <int MB>
__device__ __forceinline__ void ldx(float* v, const uint16_t* p) {
    if constexpr (MB == 8) {
        const uint4 q = __ldg(reinterpret_cast<const uint4*>(p)); const uint32_t u[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) { v[2 * k] = __uint_as_float(u[k] << 16); v[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u); }
    } else if constexpr (MB == 4) {
        const uint2 q = __ldg(reinterpret_cast<const uint2*>(p)); v[0] = __uint_as_float(q.x << 16); v[1] = __uint_as_float(q.x & 0xffff0000u);
        v[2] = __uint_as_float(q.y << 16); v[3] = __uint_as_float(q.y & 0xffff0000u);
    } else {
        const uint32_t q = __ldg(reinterpret_cast<const uint32_t*>(p)); v[0] = __uint_as_float(q << 16); v[1] = __uint_as_float(q & 0xffff0000u);
    }
}
template <int MB, typename XT>
__global__ void __launch_bounds__(32 * WPB) qvb8r_multi(const XT* __restrict__ X, const uint32_t* __restrict__ words,
    const int64_t* __restrict__ tile_off, const uint2* __restrict__ states, const int64_t* __restrict__ tiles,
    const int32_t* __restrict__ blocks, const uint32_t* __restrict__ tab, const int32_t* __restrict__ perm,
    const float* __restrict__ mn, const float* __restrict__ ts, const int32_t* __restrict__ drow, float* __restrict__ Y, int P, int R) {
    __shared__ uint32_t st[4096];
    __shared__ uint32_t ring[WPB][RING];
    BLOCK_PROLOGUE(WPB)
    __syncthreads();
    if (w >= ntb) return;
    const int tile = tb + w;
    const int r0 = (int)tiles[3 * tile + 1], nr = (int)tiles[3 * tile + 2];
    uint32_t* rg = ring[w];
    Stream z; stream_init(z, rg, words, tile_off, states, tile, lane);
    __syncwarp();
    for (int r = 0; r < nr; ++r) {
        const int orow = perm[r0 + r];
        const float m = mn[orow], t = ts[orow];
        const int dr = drow[orow];
        float acc[MB], xc[MB], xn[MB], xm[MB];
        #pragma unroll
        for (int i = 0; i < MB; ++i) acc[i] = 0.f;
        ldx<MB>(xc, X + (int64_t)lane * MB);
        for (int s0 = 0; s0 < S; s0 += 2) {
            ldx<MB>(xn, X + (int64_t)((s0 + 1) * 32 + lane) * MB);
            if (s0 + 2 < S) ldx<MB>(xm, X + (int64_t)((s0 + 2) * 32 + lane) * MB);
            uint32_t s1, s2; stream_pair(z, st, rg, lane, lt, s1, s2);
            const float w1 = fmaf((float)((int)s1 + dr), t, m), w2 = fmaf((float)((int)s2 + dr), t, m);
            #pragma unroll
            for (int i = 0; i < MB; ++i) { acc[i] = fmaf(xc[i], w1, acc[i]); acc[i] = fmaf(xn[i], w2, acc[i]); xc[i] = xm[i]; }
        }
        #pragma unroll
        for (int i = 0; i < MB; ++i) {
            #pragma unroll
            for (int o = 16; o; o >>= 1) acc[i] += __shfl_down_sync(0xffffffff, acc[i], o);
        }
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < MB; ++i) Y[(int64_t)i * R + orow] = acc[i];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// M = 9..16: warp w decodes tile tb + w; all 16 warps step in lock-step. Every TC_SB steps the 16 x 32*TC_SB fp16
// weight tile (fp16(band), as QVB8's mma path) is multiplied with the X^T tile by 2*TC_SB WMMA m16n16k16.
__global__ void __launch_bounds__(32 * TC_WARPS) qvb8r_tc(const half* __restrict__ X, const uint32_t* __restrict__ words,
    const int64_t* __restrict__ tile_off, const uint2* __restrict__ states, const int64_t* __restrict__ tiles,
    const int32_t* __restrict__ blocks, const uint32_t* __restrict__ tab, const int32_t* __restrict__ perm,
    const float* __restrict__ mn, const float* __restrict__ ts, const int32_t* __restrict__ drow, float* __restrict__ Y, int P, int R, int M, int T) {
    constexpr int CW = 32 * TC_SB;
    extern __shared__ uint32_t st[];                      // 4096-entry slot table (dynamic: static smem caps at 48 KB)
    __shared__ uint32_t ring[TC_WARPS][RING];
    __shared__ __align__(32) half Ws[2][16][CW];
    __shared__ __align__(32) half Xs[2][16][CW];
    __shared__ int orow_s[16];
    float* Yp = reinterpret_cast<float*>(&Ws[0][0][0]);   // 2*TC_SB partial 16x16 accumulators, reuses Ws at row end
    static_assert(2 * TC_SB * 256 * 4 <= 2 * 16 * CW * 2, "Yp must fit in Ws");
    BLOCK_PROLOGUE(TC_WARPS)
    for (int i = threadIdx.x; i < 2 * 16 * CW; i += blockDim.x) (&Xs[0][0][0])[i] = __float2half(0.f);
    const bool has = w < ntb;
    const int tile = tb + (has ? w : 0);
    const int r0 = (int)tiles[3 * tile + 1], nr = has ? (int)tiles[3 * tile + 2] : 0;
    uint32_t* rg = ring[w];
    Stream z;
    if (has) stream_init(z, rg, words, tile_off, states, tile, lane);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
    __syncthreads();
    for (int r = 0; r < T; ++r) {
        const bool act = r < nr;
        const int orow = act ? perm[r0 + r] : 0;
        const float m = act ? mn[orow] : 0.f, t = act ? ts[orow] : 0.f;
        const int dr = act ? drow[orow] : 0;
        if (lane == 0) orow_s[w] = act ? orow : -1;
        wmma::fill_fragment(acc, 0.f);
        half xn[TC_SB];
        #pragma unroll
        for (int j = 0; j < TC_SB; ++j) xn[j] = (w < M) ? X[(int64_t)w * P + 32 * j + lane] : __float2half(0.f);
        for (int s0 = 0, it = 0; s0 < S; s0 += TC_SB, ++it) {
            const int b = it & 1;
            if (w < M) {
                #pragma unroll
                for (int j = 0; j < TC_SB; ++j) {
                    Xs[b][w][32 * j + lane] = xn[j];
                    if (s0 + TC_SB < S) xn[j] = X[(int64_t)w * P + (s0 + TC_SB + j) * 32 + lane];
                }
            }
            #pragma unroll
            for (int j = 0; j < TC_SB; j += 2) {
                uint32_t s1 = 0, s2 = 0;
                if (act) stream_pair(z, st, rg, lane, lt, s1, s2);           // act is warp-uniform
                Ws[b][w][32 * j + lane] = act ? __float2half_rn(fmaf((float)((int)s1 + dr), t, m)) : __float2half(0.f);
                Ws[b][w][32 * (j + 1) + lane] = act ? __float2half_rn(fmaf((float)((int)s2 + dr), t, m)) : __float2half(0.f);
            }
            __syncthreads();
            if (w < 2 * TC_SB) {
                wmma::load_matrix_sync(fa, &Ws[b][0][16 * w], CW);
                wmma::load_matrix_sync(fb, &Xs[b][0][16 * w], CW);
                wmma::mma_sync(acc, fa, fb, acc);
            }
        }
        __syncthreads();
        if (w < 2 * TC_SB) wmma::store_matrix_sync(Yp + w * 256, acc, 16, wmma::mem_row_major);
        __syncthreads();
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            const int rr = i >> 4, mm = i & 15;
            if (mm < M && orow_s[rr] >= 0) {
                float v = 0.f;
                #pragma unroll
                for (int k = 0; k < 2 * TC_SB; ++k) v += Yp[k * 256 + i];
                Y[(int64_t)mm * R + orow_s[rr]] = v;
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// dense rows of blocks [b0, b0 + gridDim.x). row_base < 0: row written at its ORIGINAL index (whole weight);
// row_base >= 0: at its sorted position - row_base (a contiguous block range fills a compact chunk).
template <typename OutT> __device__ __forceinline__ OutT cvt(float v);
template <> __device__ __forceinline__ half cvt<half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 cvt<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template <typename OutT>
__global__ void __launch_bounds__(32 * WPB) qvb8r_decode(const uint32_t* __restrict__ words, const int64_t* __restrict__ tile_off,
    const uint2* __restrict__ states, const int64_t* __restrict__ tiles, const int32_t* __restrict__ blocks_all,
    const uint32_t* __restrict__ tab, const int32_t* __restrict__ perm, const float* __restrict__ mn,
    const float* __restrict__ ts, const int32_t* __restrict__ drow, OutT* __restrict__ W, int P, int b0, int64_t row_base) {
    __shared__ uint32_t st[4096];
    __shared__ uint32_t ring[WPB][RING];
    const int32_t* blocks = blocks_all + 3 * b0;
    BLOCK_PROLOGUE(WPB)
    __syncthreads();
    if (w >= ntb) return;
    const int tile = tb + w;
    const int r0 = (int)tiles[3 * tile + 1], nr = (int)tiles[3 * tile + 2];
    uint32_t* rg = ring[w];
    Stream z; stream_init(z, rg, words, tile_off, states, tile, lane);
    __syncwarp();
    for (int r = 0; r < nr; ++r) {
        const int orow = perm[r0 + r];
        const float m = mn[orow], t = ts[orow];
        const int dr = drow[orow];
        OutT* Wr = W + (row_base < 0 ? (int64_t)orow : (int64_t)(r0 + r) - row_base) * P;
        for (int s0 = 0; s0 < S; s0 += 2) {
            uint32_t s1, s2; stream_pair(z, st, rg, lane, lt, s1, s2);
            Wr[s0 * 32 + lane] = cvt<OutT>(fmaf((float)((int)s1 + dr), t, m));
            Wr[(s0 + 1) * 32 + lane] = cvt<OutT>(fmaf((float)((int)s2 + dr), t, m));
        }
    }
}

// One thread per (m, output q that has escapes): its escapes are summed in their stored (sorted) order, then added
// once. No atomics: the result is bit-reproducible (an atomicAdd per escape added a row's escapes in whatever order
// the warps ran -- several escapes share an output -- so the fp32 sum, and after rounding the output, changed run to run).
__global__ void qvb8r_fix_escapes(const float* __restrict__ X, const int64_t* __restrict__ uq, const int64_t* __restrict__ ptr,
    const int64_t* __restrict__ ec, const float* __restrict__ ed, float* __restrict__ Y, int64_t U, int M, int P, int Q) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= U * M) return;
    const int64_t u = i % U; const int m = (int)(i / U);
    const float* x = X + (int64_t)m * P;
    float a = 0.f;
    for (int64_t e = ptr[u]; e < ptr[u + 1]; ++e) a = fmaf(x[ec[e]], ed[e], a);
    Y[(int64_t)m * Q + uq[u]] += a;
}

// ---------------------------------------------------------------------------------------------------------------
// host wrappers. "core" = words (u32 as i32), tile_off i64, states (u64 as i64), tiles i64, blocks i32,
// tab (i32 view), perm i32, min_s f32, tau_s f32, drow i32 -- all on one CUDA device.
#define CORE_ARGS (const uint32_t*)words.data_ptr<int32_t>(), off.data_ptr<int64_t>(), \
    (const uint2*)states.data_ptr<int64_t>(), tiles.data_ptr<int64_t>(), blocks.data_ptr<int32_t>(), \
    (const uint32_t*)tab.data_ptr<int32_t>(), perm.data_ptr<int32_t>(), mn.data_ptr<float>(), ts.data_ptr<float>(), \
    drow.data_ptr<int32_t>()
#define CORE_PARAMS torch::Tensor words, torch::Tensor off, torch::Tensor states, torch::Tensor tiles, \
    torch::Tensor blocks, torch::Tensor tab, torch::Tensor perm, torch::Tensor mn, torch::Tensor ts, torch::Tensor drow
#define CORE_NAMES words, off, states, tiles, blocks, tab, perm, mn, ts, drow

// Every entry point runs on the card its tensors live on, not whichever card is current (pipeline placement
// spreads layers over cards; launching on the wrong one reads foreign pointers -> Xid 31).
#define DEVICE_GUARD(t) const at::cuda::OptionalCUDAGuard _guard(at::device_of(t))

static void check_core(CORE_PARAMS) {
    CHECK_CUDA_INPUTS(CORE_NAMES);
    TORCH_CHECK(words.scalar_type() == torch::kInt32 && off.scalar_type() == torch::kInt64 &&
                states.scalar_type() == torch::kInt64 && tiles.scalar_type() == torch::kInt64 &&
                blocks.scalar_type() == torch::kInt32 && tab.scalar_type() == torch::kInt32 &&
                perm.scalar_type() == torch::kInt32 && mn.scalar_type() == torch::kFloat &&
                ts.scalar_type() == torch::kFloat && drow.scalar_type() == torch::kInt32, "qvb8r: core tensor dtypes");
}

torch::Tensor gemv(torch::Tensor X, CORE_PARAMS) {
    DEVICE_GUARD(words);
    check_core(CORE_NAMES);
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat && X.is_contiguous() && X.numel() % 128 == 0,
                "qvb8r gemv: X must be contiguous fp32 CUDA, P % 128 == 0");
    const int P = X.numel(), R = mn.numel();
    auto Y = torch::empty({R}, X.options());
    qvb8r_gemv<false><<<blocks.numel() / 3, 32 * WPB, 0, at::cuda::getCurrentCUDAStream()>>>(X.data_ptr<float>(), CORE_ARGS, Y.data_ptr<float>(), nullptr, P);
    return Y;
}

// every id of the tensor, (Q, P) int16, original row order -- for verifying a written file
torch::Tensor ids(int64_t P, CORE_PARAMS) {
    DEVICE_GUARD(words);
    check_core(CORE_NAMES);
    TORCH_CHECK(P % 128 == 0, "qvb8r ids: P % 128 == 0");
    const int R = mn.numel();
    auto X = torch::zeros({P}, mn.options());
    auto Y = torch::empty({R}, mn.options());
    auto D = torch::empty({R, P}, mn.options().dtype(torch::kInt16));
    qvb8r_gemv<true><<<blocks.numel() / 3, 32 * WPB, 0, at::cuda::getCurrentCUDAStream()>>>(X.data_ptr<float>(), CORE_ARGS, Y.data_ptr<float>(),
                                                       D.data_ptr<int16_t>(), (int)P);
    return D;
}

torch::Tensor multi(torch::Tensor X, CORE_PARAMS) {
    DEVICE_GUARD(words);
    check_core(CORE_NAMES);
    const bool bf = X.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(X.is_cuda() && (bf || X.scalar_type() == torch::kFloat) && X.dim() == 2,
                "qvb8r multi: X must be (M, P) fp32 or bf16 CUDA");
    const int M = X.size(0), P = X.size(1), R = mn.numel();
    TORCH_CHECK(M >= 1 && M <= 8 && P % 128 == 0, "qvb8r multi: M must be 1..8 and P % 128 == 0");
    const int MP = M <= 2 ? 2 : M <= 4 ? 4 : 8;            // instantiated batch, extra rows zero
    auto Xt = torch::zeros({P, MP}, X.options());
    Xt.narrow(1, 0, M).copy_(X.t());
    auto Y = torch::empty({MP, R}, X.options().dtype(torch::kFloat));
    const int nb = blocks.numel() / 3;
    auto go = [&](auto kern, const auto* xp) {
        kern<<<nb, 32 * WPB, 0, at::cuda::getCurrentCUDAStream()>>>(xp, CORE_ARGS, Y.data_ptr<float>(), P, R); };
    if (bf) {
        const uint16_t* xp = (const uint16_t*)Xt.data_ptr<at::BFloat16>();
        if (MP == 2) go(qvb8r_multi<2, uint16_t>, xp); else if (MP == 4) go(qvb8r_multi<4, uint16_t>, xp); else go(qvb8r_multi<8, uint16_t>, xp);
    } else {
        const float* xp = Xt.data_ptr<float>();
        if (MP == 2) go(qvb8r_multi<2, float>, xp); else if (MP == 4) go(qvb8r_multi<4, float>, xp); else go(qvb8r_multi<8, float>, xp);
    }
    return Y.narrow(0, 0, M);
}

torch::Tensor tc(torch::Tensor X, CORE_PARAMS, int64_t T) {
    DEVICE_GUARD(words);
    check_core(CORE_NAMES);
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kHalf && X.is_contiguous() && X.dim() == 2,
                "qvb8r tc: X must be contiguous (M, P) fp16 CUDA");
    const int M = X.size(0), P = X.size(1), R = mn.numel();
    TORCH_CHECK(M >= 1 && M <= 16 && P % (32 * TC_SB) == 0, "qvb8r tc: M must be 1..16 and P % 128 == 0");
    auto Y = torch::empty({M, R}, X.options().dtype(torch::kFloat));
    cudaFuncSetAttribute(qvb8r_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096 * 4);
    qvb8r_tc<<<blocks.numel() / 3, 32 * TC_WARPS, 4096 * 4, at::cuda::getCurrentCUDAStream()>>>((const half*)X.data_ptr<at::Half>(), CORE_ARGS,
                                                             Y.data_ptr<float>(), P, R, M, (int)T);
    return Y;
}

// out dtype: 0 fp16, 1 bf16, 2 fp32. b0/nb: block range; row_base < 0 -> whole (Q, P) weight in original row order,
// else a (nrows, P) chunk in sorted order.
torch::Tensor decode(CORE_PARAMS, int64_t P, int64_t out_dtype, int64_t b0, int64_t nb, int64_t row_base, int64_t nrows) {
    DEVICE_GUARD(words);
    check_core(CORE_NAMES);
    TORCH_CHECK(P % 128 == 0, "qvb8r decode: P % 128 == 0");
    TORCH_CHECK(b0 >= 0 && nb >= 0 && b0 + nb <= blocks.numel() / 3, "qvb8r decode: block range out of bounds");
    const int64_t rows = row_base < 0 ? mn.numel() : nrows;
    const auto dt = out_dtype == 0 ? torch::kHalf : out_dtype == 1 ? torch::kBFloat16 : torch::kFloat;
    auto W = torch::empty({rows, P}, mn.options().dtype(dt));
    if (nb == 0) return W;
    auto go = [&](auto kern, auto* out) { kern<<<(int)nb, 32 * WPB, 0, at::cuda::getCurrentCUDAStream()>>>(CORE_ARGS,
        out, (int)P, (int)b0, row_base); };
    if (out_dtype == 0) go(qvb8r_decode<half>, (half*)W.data_ptr<at::Half>());
    else if (out_dtype == 1) go(qvb8r_decode<__nv_bfloat16>, (__nv_bfloat16*)W.data_ptr<at::BFloat16>());
    else go(qvb8r_decode<float>, W.data_ptr<float>());
    return W;
}

void fix_escapes(torch::Tensor X, torch::Tensor uq, torch::Tensor ptr, torch::Tensor ec, torch::Tensor ed, torch::Tensor Y) {
    DEVICE_GUARD(Y);
    TORCH_CHECK(X.scalar_type() == torch::kFloat && Y.scalar_type() == torch::kFloat && X.is_contiguous() && Y.is_contiguous(),
                "qvb8r fix_escapes: X and Y must be contiguous fp32");
    const int64_t U = uq.numel(); const int M = X.size(0), P = X.size(1), Q = Y.size(1);
    if (U == 0) return;
    const int64_t n = U * M;
    qvb8r_fix_escapes<<<(n + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(X.data_ptr<float>(), uq.data_ptr<int64_t>(),
        ptr.data_ptr<int64_t>(), ec.data_ptr<int64_t>(), ed.data_ptr<float>(), Y.data_ptr<float>(), U, M, P, Q);
}

torch::Tensor embed_rows(torch::Tensor rows, torch::Tensor words, torch::Tensor off, torch::Tensor states,
    torch::Tensor tiles, torch::Tensor tab, torch::Tensor inv_perm, torch::Tensor row_tile, torch::Tensor mn,
    torch::Tensor ts, torch::Tensor drow, torch::Tensor esc_ptr, torch::Tensor esc_col, torch::Tensor esc_val, int64_t P);
std::vector<torch::Tensor> encode(torch::Tensor sym, torch::Tensor tiles, torch::Tensor F, torch::Tensor C);
torch::Tensor row_hist(torch::Tensor sym, int64_t N);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv", &gemv, "M = 1 GEMV (fp32 X)");
    m.def("ids", &ids, "all ids (Q, P) int16, for verification");
    m.def("multi", &multi, "M = 2..8 (fp32 or bf16 X)");
    m.def("tc", &tc, "M = 9..16 tensor-core path (fp16 X)");
    m.def("decode", &decode, "dense rows");
    m.def("fix_escapes", &fix_escapes, "escape correction");
    m.def("embed_rows", &embed_rows, "CPU embedding rows");
    m.def("encode", &encode, "CPU rANS encoder (codec.encode_tensor)");
    m.def("row_hist", &row_hist, "CPU per-row symbol histograms (codec)");
}
