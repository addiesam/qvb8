// QVB8 GEMV -- per-row virtual-band weight codec.
//
//   y[q] = sum_c X[c] * (min_s[q] + id[q][c] * tau_s)
//        = min_s[q] * sum(X) + tau_s * sum_c X[c]*id[q][c]
//
// so the per-weight work is ONE fma on the raw index: the row origin and
// the bucket step both factor out of the loop into two scalar operations
// in the epilogue. There is no codebook to gather -- entry i of the
// virtual codebook is min_s + i*tau_s by arithmetic -- which is the whole
// point of the format. the earlier format's anchored kernel pays a data-dependent
// `codebook[id]` shared-memory load per weight, and that gather is what
// holds it to 38-66% of this box's memory bandwidth.
//
// DELIBERATELY INDEPENDENT of the the earlier format backend: separate translation
// unit, separate extension module, no shared headers. The formats have
// different invariants and must be able to change without reasoning about
// each other.
//
// Row widths VARY (each row stores ceil(log2(its own bucket count)) bits),
// so the kernel reads row_nbits[q] and row_bit_off[q] rather than assuming
// a global stride. Within a row the width is uniform, so a lane still
// strides uniformly once it has the row's base -- which is what keeps the
// index loads coalesced despite the ragged rows.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDABlas.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <cstdlib>
#include <cstring>  // strcmp, for the QVB8_GEMM_OUT env lookup
#include <unordered_map>

#define WARP 32

// An index is at most MAX_CERTIFIED_NBITS (14) bits and starts at an
// arbitrary bit offset, so it spans at most 3 bytes. Reading 4 gives a
// whole aligned word and costs nothing extra; the packer pads the buffer
// so this never runs off the end.
__device__ __forceinline__ uint32_t load_field(const uint8_t* __restrict__ ids,
                                               int64_t bit, int nbits) {
    int64_t byi = bit >> 3;
    uint32_t sh = (uint32_t)(bit & 7);
    uint32_t w = (uint32_t)ids[byi]
               | ((uint32_t)ids[byi + 1] << 8)
               | ((uint32_t)ids[byi + 2] << 16)
               | ((uint32_t)ids[byi + 3] << 24);
    return (w >> sh) & ((1u << nbits) - 1u);
}


// Small-M (1..8) GEMV: the speculative-verify / small-prefill shape. Decodes
// each weight ONCE and multiplies it into MB accumulators, so the cost is the
// same one pass over the packed ids that M=1 makes -- as against decoding the
// whole (Q, P) matrix to hand a GEMM three rows of work, which is a fixed cost
// independent of M and measured 3.2x slower than the earlier format at M=3.
//
// MB is a compile-time REGISTER-BLOCK size and the launch picks the
// instantiation whose MB equals the caller's M (see
// qvb8_gemv_multi_forward); the kernel additionally bounds every X read and
// Y write by the runtime M, so a future dispatch mistake degrades to wrong
// arithmetic rather than a heap overflow.


// ---------------------------------------------------------------------------
// v2 kernels: same arithmetic, more bytes in flight.
//
// The v1 loops (removed 2026-10-12) issued ONE dependent load per warp per iteration -- about
// 4*nbits bytes (32 B at 8 bits) -- and then shuffle and FMA on it before the
// next load is even issued. On this box (~2300 resident warps, ~700 ns DRAM
// latency) Little's law caps that at roughly 2300 * 32 B / 700 ns ~ 100 GB/s,
// which is what the profiler measured (a 27B forward reading ~27 GB in
// ~300 ms, i.e. ~90 GB/s of the ~270 GB/s the part can do). The loop is
// latency-bound, not bandwidth-bound.
//
// Fix: a warp fetches the raw words for QVB8_U consecutive iterations first
// (independent loads, all in flight together) and only then runs the
// shuffle/FMA phase. Per-lane element order is unchanged, so every result is
// bit-identical to v1. The four single-byte loads per lane are also replaced
// by two aligned 32-bit loads and a funnel shift.
//
// Requires a 4-byte-aligned `ids` base (the host enforces it)
// and 64 bytes of reach past a warp's first byte; an iteration closer to the
// buffer's end than that takes the per-lane load_field.
#ifndef QVB8_U
#define QVB8_U 4
#endif
#define QVB8_REACH 64
// Words a warp fetches per iteration: 32 lanes x <= 14 bits spans <= 448
// bits = 14 words, +1 for the unaligned start. Lanes past it re-read word 0.
#define QVB8_WARP_WORDS 15

__device__ __forceinline__ uint32_t fetch_word_v2(const uint8_t* __restrict__ ids,
                                                  int64_t warp_byte_base, int lane) {
    const int fetch_lane = lane < QVB8_WARP_WORDS ? lane : 0;
    const int64_t a = warp_byte_base + (int64_t)fetch_lane * 4;
    const int64_t a0 = a & ~(int64_t)3;
    const uint32_t w0 = __ldg(reinterpret_cast<const uint32_t*>(ids + a0));
    const uint32_t w1 = __ldg(reinterpret_cast<const uint32_t*>(ids + a0 + 4));
    return __funnelshift_r(w0, w1, (uint32_t)(a & 3) * 8);
}

__device__ __forceinline__ uint32_t extract_field_v2(uint32_t my_word,
                                                     int64_t warp_bit_base,
                                                     int lane, int nbits) {
    const uint32_t field_bit = (uint32_t)lane * (uint32_t)nbits
                             + (uint32_t)(warp_bit_base & 7);
    const int word_idx = (int)(field_bit >> 5);
    const uint32_t bit_in_word = field_bit & 31u;
    const uint32_t lo = __shfl_sync(0xffffffff, my_word, word_idx);
    const uint32_t hi = __shfl_sync(0xffffffff, my_word, word_idx + 1);
    const uint64_t combined = (uint64_t)lo | ((uint64_t)hi << 32);
    return (uint32_t)(combined >> bit_in_word) & ((1u << nbits) - 1u);
}

template <int BW, int MB>
__global__ void __launch_bounds__(BW*WARP) qvb8_gemv_multi_v2(
    const half*    __restrict__ X,          // (M, P)
    const uint8_t* __restrict__ ids,
    const float*   __restrict__ min_s,
    const uint8_t* __restrict__ row_nbits,
    const int64_t* __restrict__ row_bit_off,
    float*         __restrict__ Y,          // (M, Q)
    const float*   __restrict__ tau_s,     // (Q)
    int P, int Q, int M, int64_t ids_bytes) {

    int warp = threadIdx.x / WARP;
    int lane = threadIdx.x % WARP;
    int q = blockIdx.x * BW + warp;
    if (q >= Q) return;

    const int nb = row_nbits[q];
    const int64_t base = row_bit_off[q];
    const float m = min_s[q];
    const float ts = tau_s[q];
    float acc[MB];
    #pragma unroll
    for (int i = 0; i < MB; ++i) acc[i] = 0.f;

    // Split over P: blockIdx.y selects a contiguous range of 32-element
    // iterations. gridDim.y == 1 is the whole row. Partial sums go to
    // Y + y*M*Q and are combined in fixed order by qvb8_splitk_combine.
    const int n_iter = (P + WARP - 1) / WARP;
    const int per = (n_iter + (int)gridDim.y - 1) / (int)gridDim.y;
    const int i_begin = (int)blockIdx.y * per;
    const int i_end = min(n_iter, i_begin + per);
    Y += (size_t)blockIdx.y * M * Q;

    for (int i0 = i_begin; i0 < i_end; i0 += QVB8_U) {
        uint32_t raw[QVB8_U];
        bool ok[QVB8_U];
        #pragma unroll
        for (int u = 0; u < QVB8_U; ++u) {
            const int i = i0 + u;
            const int64_t wbb = (base + (int64_t)(i * WARP) * nb) >> 3;
            ok[u] = (i < i_end) && (ids_bytes - wbb) >= QVB8_REACH;
            raw[u] = ok[u] ? fetch_word_v2(ids, wbb, lane) : 0u;
        }
        #pragma unroll
        for (int u = 0; u < QVB8_U; ++u) {
            const int i = i0 + u;
            if (i >= i_end) break;                    // warp-uniform
            const int c = i * WARP + lane;
            const int safe_lane = (c < P) ? lane : 0;
            const int64_t wbits = base + (int64_t)(i * WARP) * nb;
            const uint32_t id = ok[u]
                ? extract_field_v2(raw[u], wbits, safe_lane, nb)
                : load_field(ids, base + (int64_t)(i * WARP + safe_lane) * nb, nb);
            if (c < P) {
                const float w = fmaf((float)id, ts, m);
                #pragma unroll
                for (int i2 = 0; i2 < MB; ++i2) {
                    if (i2 < M)
                        acc[i2] = fmaf(__half2float(__ldg(X + (size_t)i2 * P + c)), w, acc[i2]);
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < MB; ++i) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            acc[i] += __shfl_down_sync(0xffffffff, acc[i], o);
    }
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < MB; ++i) {
            if (i < M) Y[(int64_t)i * Q + q] = acc[i];
        }
    }
}

// M=1. XS=true stages X in shared memory (small P), false reads it via __ldg
// (large P); qvb8_gemv_forward picks by QVB8_GEMV_SMEM_MAX_P.
template <int BW, bool XS>
__global__ void __launch_bounds__(BW*WARP) qvb8_gemv_v2(
    const half*    __restrict__ X,          // (P)
    const uint8_t* __restrict__ ids,
    const float*   __restrict__ min_s,
    const uint8_t* __restrict__ row_nbits,
    const int64_t* __restrict__ row_bit_off,
    float*         __restrict__ Y,          // (Q)
    const float*   __restrict__ tau_s,
    int P, int Q, int64_t ids_bytes) {

    extern __shared__ unsigned char smem[];
    half* Xs = reinterpret_cast<half*>(smem);
    if (XS) {
        for (int c = threadIdx.x; c < P; c += blockDim.x) Xs[c] = X[c];
        __syncthreads();
    }
    int warp = threadIdx.x / WARP;
    int lane = threadIdx.x % WARP;
    int q = blockIdx.x * BW + warp;
    if (q >= Q) return;

    const int nb = row_nbits[q];
    const int64_t base = row_bit_off[q];
    const float m = min_s[q];
    const float ts = tau_s[q];
    float acc = 0.f;

    const int n_iter = (P + WARP - 1) / WARP;
    const int per = (n_iter + (int)gridDim.y - 1) / (int)gridDim.y;
    const int i_begin = (int)blockIdx.y * per;
    const int i_end = min(n_iter, i_begin + per);
    Y += (size_t)blockIdx.y * Q;

    for (int i0 = i_begin; i0 < i_end; i0 += QVB8_U) {
        uint32_t raw[QVB8_U];
        bool ok[QVB8_U];
        #pragma unroll
        for (int u = 0; u < QVB8_U; ++u) {
            const int i = i0 + u;
            const int64_t wbb = (base + (int64_t)(i * WARP) * nb) >> 3;
            ok[u] = (i < i_end) && (ids_bytes - wbb) >= QVB8_REACH;
            raw[u] = ok[u] ? fetch_word_v2(ids, wbb, lane) : 0u;
        }
        #pragma unroll
        for (int u = 0; u < QVB8_U; ++u) {
            const int i = i0 + u;
            if (i >= i_end) break;
            const int c = i * WARP + lane;
            const int safe_lane = (c < P) ? lane : 0;
            const int64_t wbits = base + (int64_t)(i * WARP) * nb;
            const uint32_t id = ok[u]
                ? extract_field_v2(raw[u], wbits, safe_lane, nb)
                : load_field(ids, base + (int64_t)(i * WARP + safe_lane) * nb, nb);
            if (c < P) {
                const float w = fmaf((float)id, ts, m);
                const float x = XS ? __half2float(Xs[c]) : __half2float(__ldg(X + c));
                acc = fmaf(x, w, acc);
            }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) Y[q] = acc;
}

// ---------------------------------------------------------------------------
// v3 kernel: lane-contiguous fields, no warp shuffles.
//
// v2 still assigns one ELEMENT per lane per iteration and redistributes the
// packed bits with two shuffles per element, so a weight costs ~30 instructions
// and each warp-iteration moves only ~4*nbits bytes. The real packs are almost
// entirely 6- and 7-bit rows (80% / 19% of weights), which allows a layout that
// needs none of that: a lane owns EIGHT consecutive elements, i.e. exactly NB
// consecutive BYTES of the row (8*NB bits, byte aligned because P % 8 == 0). Its
// eight fields are then extracted from one 64-bit window with compile-time
// shifts, and its eight activations arrive as ONE 16-byte vector load. A warp
// covers 256 elements = 32*NB contiguous bytes per iteration (192/224 B).
//
// Rows of any other width take the v2-style loop inside the same kernel, so one
// launch handles a mixed pack. Preconditions (checked on the host, else v2):
// P % 8 == 0, 4-byte-aligned ids, 16-byte-aligned X rows.
#ifndef QVB8_V3_U
#define QVB8_V3_U 4
#endif
// Exact int -> float for 0 <= i < 2^23 without the (slow-pipe) I2F: OR the
// integer into the mantissa of 2^23, then subtract 2^23. Both steps are exact.
__device__ __forceinline__ float small_uint_to_float(uint32_t i) {
    return __uint_as_float(i | 0x4B000000u) - 8388608.0f;
}

// Field j (NB bits) of the 64-bit window (hi:lo), with every shift resolved at
// compile time so no 64-bit shift is emitted.
template <int NB, int J>
__device__ __forceinline__ uint32_t field_of(uint32_t lo, uint32_t hi) {
    constexpr unsigned MASK = (1u << NB) - 1u;
    constexpr int S = J * NB;
    if constexpr (S + NB <= 32)      return (lo >> S) & MASK;
    else if constexpr (S >= 32)      return (hi >> (S - 32)) & MASK;
    else                              return __funnelshift_r(lo, hi, S) & MASK;
}

#ifndef QVB8_V3_U
#define QVB8_V3_U 4
#endif
template <int MB, int NB>
__device__ __forceinline__ void row_v3_fixed(
    const float* __restrict__ X, const uint8_t* __restrict__ ids,
    int64_t base_byte, int P, int M, float m, float ts, float (&acc)[MB],
    int lane, int blk_begin, int blk_end) {
    constexpr int U = QVB8_V3_U;
    // X is the activation already converted to fp32 (see the launcher): the
    // half -> float conversion was repeated for every output row and ran on
    // the slow conversion pipe, which -- not memory, not issue slots -- was
    // what bounded the v3 kernel (ablating the X or weight loads changed
    // nothing; the time followed the conversion count).
    for (int b0 = blk_begin; b0 < blk_end; b0 += U) {
        // Only the weight words stream from DRAM, so only they are prefetched
        // across the U blocks; X is small and L1-resident and is loaded where it
        // is used (preloading it for U blocks x MB rows needed ~100 registers and
        // spilled).
        uint32_t w[U][3];
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            const int blk = b0 + u;
            const int c0 = blk * 256 + lane * 8;
            const bool valid = blk < blk_end;
            const bool live = valid && c0 < P;
            const int64_t A = base_byte + ((int64_t)(valid ? blk : b0) * 32 + (live ? lane : 0)) * NB;
            const int64_t a0 = A & ~(int64_t)3;
            w[u][0] = __ldg(reinterpret_cast<const uint32_t*>(ids + a0));
            w[u][1] = __ldg(reinterpret_cast<const uint32_t*>(ids + a0 + 4));
            w[u][2] = __ldg(reinterpret_cast<const uint32_t*>(ids + a0 + 8));
        }
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            const int blk = b0 + u;
            if (blk >= blk_end) break;                          // warp-uniform
            const int c0 = blk * 256 + lane * 8;
            const bool live = c0 < P;
            const int64_t A = base_byte + ((int64_t)blk * 32 + (live ? lane : 0)) * NB;
            const uint32_t sh = (uint32_t)(A & 3) * 8u;
            const uint32_t lo = __funnelshift_r(w[u][0], w[u][1], sh);
            const uint32_t hi = __funnelshift_r(w[u][1], w[u][2], sh);
            float wv[8];
            wv[0] = fmaf(small_uint_to_float(field_of<NB, 0>(lo, hi)), ts, m);
            wv[1] = fmaf(small_uint_to_float(field_of<NB, 1>(lo, hi)), ts, m);
            wv[2] = fmaf(small_uint_to_float(field_of<NB, 2>(lo, hi)), ts, m);
            wv[3] = fmaf(small_uint_to_float(field_of<NB, 3>(lo, hi)), ts, m);
            wv[4] = fmaf(small_uint_to_float(field_of<NB, 4>(lo, hi)), ts, m);
            wv[5] = fmaf(small_uint_to_float(field_of<NB, 5>(lo, hi)), ts, m);
            wv[6] = fmaf(small_uint_to_float(field_of<NB, 6>(lo, hi)), ts, m);
            wv[7] = fmaf(small_uint_to_float(field_of<NB, 7>(lo, hi)), ts, m);
            #pragma unroll
            for (int r = 0; r < MB; ++r) {
                float4 xa = make_float4(0.f, 0.f, 0.f, 0.f), xb = xa;
                if (live && r < M) {
                    const float4* xp = reinterpret_cast<const float4*>(X + (size_t)r * P + c0);
                    xa = __ldg(xp);
                    xb = __ldg(xp + 1);
                }
                acc[r] = fmaf(xa.x, wv[0], acc[r]);
                acc[r] = fmaf(xa.y, wv[1], acc[r]);
                acc[r] = fmaf(xa.z, wv[2], acc[r]);
                acc[r] = fmaf(xa.w, wv[3], acc[r]);
                acc[r] = fmaf(xb.x, wv[4], acc[r]);
                acc[r] = fmaf(xb.y, wv[5], acc[r]);
                acc[r] = fmaf(xb.z, wv[6], acc[r]);
                acc[r] = fmaf(xb.w, wv[7], acc[r]);
            }
        }
    }
}

template <int BW, int MB>
__global__ void __launch_bounds__(BW*WARP) qvb8_gemv_multi_v3(
    const float*   __restrict__ X,          // (M, P) fp32 (already narrowed to fp16 precision)
    const uint8_t* __restrict__ ids,
    const float*   __restrict__ min_s,
    const uint8_t* __restrict__ row_nbits,
    const int64_t* __restrict__ row_bit_off,
    float*         __restrict__ Y,          // (S, M, Q) partials when gridDim.y > 1
    const float*   __restrict__ tau_s,
    int P, int Q, int M, int64_t ids_bytes) {

    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int q = blockIdx.x * BW + warp;
    if (q >= Q) return;

    const int nb = row_nbits[q];
    const int64_t base_bit = row_bit_off[q];
    const float m = min_s[q];
    const float ts = tau_s[q];
    float acc[MB];
    #pragma unroll
    for (int i = 0; i < MB; ++i) acc[i] = 0.f;

    const int n_blk = (P + 255) / 256;
    const int per = (n_blk + (int)gridDim.y - 1) / (int)gridDim.y;
    const int b_begin = (int)blockIdx.y * per;
    const int b_end = min(n_blk, b_begin + per);
    Y += (size_t)blockIdx.y * M * Q;

    // Eight fields always occupy exactly NB bytes, so widths 4..8 fit the 64-bit
    // window the fixed-width path extracts from. The common packs are almost
    // entirely 6/7-bit (tau 1e-3) or 4/5-bit (tau 5e-3).
    if (nb == 6) {
        row_v3_fixed<MB, 6>(X, ids, base_bit >> 3, P, M, m, ts, acc, lane, b_begin, b_end);
    } else if (nb == 7) {
        row_v3_fixed<MB, 7>(X, ids, base_bit >> 3, P, M, m, ts, acc, lane, b_begin, b_end);
    } else if (nb == 4) {
        row_v3_fixed<MB, 4>(X, ids, base_bit >> 3, P, M, m, ts, acc, lane, b_begin, b_end);
    } else if (nb == 5) {
        row_v3_fixed<MB, 5>(X, ids, base_bit >> 3, P, M, m, ts, acc, lane, b_begin, b_end);
    } else if (nb == 8) {
        row_v3_fixed<MB, 8>(X, ids, base_bit >> 3, P, M, m, ts, acc, lane, b_begin, b_end);
    } else {
        // (3-bit rows deliberately take this path: the fixed-width window loads
        // 12 bytes per lane, which for NB=3 can run 9 bytes past the end of the
        // buffer on its last row -- more than the 8 bytes of trailing pad the
        // packer guarantees. NB >= 4 over-reads at most 8.)
        // Any other width: the v2-style per-element path over the same range
        // (256 elements = 8 of its 32-element iterations).
        const int n_iter = (P + WARP - 1) / WARP;
        const int i_begin = b_begin * 8;
        const int i_end = min(n_iter, b_end * 8);
        for (int i = i_begin; i < i_end; ++i) {
            const int c = i * WARP + lane;
            const int64_t wbits = base_bit + (int64_t)(i * WARP) * nb;
            const int safe_lane = (c < P) ? lane : 0;
            const uint32_t id = load_field(ids, base_bit + (int64_t)(i * WARP + safe_lane) * nb, nb);
            if (c < P) {
                const float w = fmaf((float)id, ts, m);
                #pragma unroll
                for (int r = 0; r < MB; ++r)
                    if (r < M) acc[r] = fmaf(__ldg(X + (size_t)r * P + c), w, acc[r]);
            }
            (void)wbits;
        }
    }
    #pragma unroll
    for (int i = 0; i < MB; ++i) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            acc[i] += __shfl_down_sync(0xffffffff, acc[i], o);
    }
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < MB; ++i)
            if (i < M) Y[(int64_t)i * Q + q] = acc[i];
    }
}

static bool qvb8_use_v3(const torch::Tensor& ids, const torch::Tensor& X, int64_t P) {
    return (P % 8) == 0
        && (reinterpret_cast<uintptr_t>(ids.data_ptr<uint8_t>()) & 3u) == 0
        && (reinterpret_cast<uintptr_t>(X.data_ptr()) & 15u) == 0;
}

// Sum S partial (M, Q) results in fixed order (s = 0..S-1), so a split GEMV
// is deterministic run to run.
__global__ void qvb8_splitk_combine(const float* __restrict__ part,
                                    float* __restrict__ out, int64_t n, int S) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = part[i];
    for (int s = 1; s < S; ++s) a += part[(size_t)s * n + i];
    out[i] = a;
}

// How many P-segments to split a GEMV into. A row is one warp, so a
// projection with few rows (down_proj: 5120 rows x 17408) launches fewer warps
// than the GPU needs to stay busy: ~2.2 waves on this part, and the last wave
// is mostly idle. Splitting each row over S warps raises the warp count to
// ~QVB8_SPLIT_WARPS. QVB8_SPLIT_WARPS=0 disables splitting.
static int qvb8_split_for(int64_t Q, int64_t P) {
    static int target = -1;
    if (target < 0) {
        const char* e = std::getenv("QVB8_SPLIT_WARPS");
        target = e ? std::atoi(e) : 20000;
    }
    if (target <= 0 || P < 4096 || Q >= target) return 1;
    int64_t s = (target + Q - 1) / Q;
    const int64_t max_by_p = (P / WARP) / (2 * QVB8_U);   // keep >= 2U iterations each
    if (s > max_by_p) s = max_by_p;
    if (s > 8) s = 8;
    return s < 1 ? 1 : (int)s;
}

// v2 is the general GEMV (any P); v3 its fast path. Both read ids as aligned
// 4-byte words. Every module owns its ids tensor (allocations are 256-byte
// aligned), so this only fires on a hand-made offset view -- the old v1
// byte-wise kernels that covered that case were removed as the slow path.
static void qvb8_check_ids_aligned(const torch::Tensor& ids) {
    TORCH_CHECK((reinterpret_cast<uintptr_t>(ids.data_ptr<uint8_t>()) & 3u) == 0,
                "qvb8 gemv: ids buffer must be 4-byte aligned (pass a "
                "contiguous copy, not an offset view)");
}

// Escapes are stored verbatim, so the banded pass above computed the WRONG
// contribution for those elements. Rather than branch inside the hot loop
// for a handful of positions, correct them afterwards: add
// (true_value - banded_value) * X[c] for each escaped element. They are
// sparse (0 of 27.8 B weights on the real 27B at tau=1e-3), so this kernel
// usually launches over an empty index list.
__global__ void qvb8_fix_escapes(
    const half*    __restrict__ X,
    const uint8_t* __restrict__ ids,
    const float*   __restrict__ min_s,
    const uint8_t* __restrict__ row_nbits,
    const int64_t* __restrict__ row_bit_off,
    const int64_t* __restrict__ esc_idx,    // flat position q*P + c
    const float*   __restrict__ esc_vals,   // the exact value
    float*         __restrict__ Y,
    const float*   __restrict__ tau_s,     // (Q)
    int P, int64_t n_esc) {
    // esc_idx is sorted (QVBandLinearV8): the thread at the first escape of an output q adds all of q's escapes in
    // order, then writes once -- no atomics, bit-reproducible (several escapes can share an output)
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_esc) return;
    const int64_t q = esc_idx[i] / P;
    if (i > 0 && esc_idx[i - 1] / P == q) return;
    float a = 0.f;
    for (int64_t e = i; e < n_esc && esc_idx[e] / P == q; ++e) {
        int c = (int)(esc_idx[e] - q * P);
        int nb = row_nbits[q];
        uint32_t id = load_field(ids, row_bit_off[q] + (int64_t)c * nb, nb);
        float banded = fmaf((float)id, tau_s[q], min_s[q]);
        a += __half2float(X[c]) * (esc_vals[e] - banded);
    }
    Y[q] += a;
}

template<typename OutT>
__global__ void qvb8_decode_all(
    const uint8_t* __restrict__ ids,
    const float*   __restrict__ min_s,
    const uint8_t* __restrict__ row_nbits,
    const int64_t* __restrict__ row_bit_off,
    OutT*          __restrict__ out,        // (Q, P)
    const float*   __restrict__ tau_s,     // (Q)
    int P, int Q) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = (int64_t)Q * P;
    if (t >= n) return;
    int64_t q = t / P;
    int c = (int)(t - q * P);
    int nb = row_nbits[q];
    uint32_t id = load_field(ids, row_bit_off[q] + (int64_t)c * nb, nb);
    out[t] = (OutT)fmaf((float)id, tau_s[q], min_s[q]);
}

// Fast full decode (2026-10-16). The per-element kernel above gives each thread ONE weight: a 64-bit
// division to find its row, four byte loads for one field, and a 2-byte store, which held a Qwen-Image
// forward's 20.4 G-weight decode to ~150 GB/s. Here a thread decodes EIGHT consecutive weights of a row:
// eight NB-bit fields are exactly NB bytes (rows are byte-aligned when P % 8 == 0, the v3 GEMV invariant),
// read as three aligned 32-bit words and unpacked with compile-time shifts (field_of), and the eight
// results leave as ONE 16-byte store. The row comes from blockIdx.y, so there is no division. Each value
// is fmaf(id, tau, min) rounded once to OutT exactly as in qvb8_decode_all, so the output is bit-identical.
// Widths outside 4..8 fall back to load_field per element inside the same thread.
template <typename OutT> struct Pack8;
template <> struct Pack8<at::BFloat16> {
    static __device__ __forceinline__ uint32_t two(float a, float b) {
        __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
        return *reinterpret_cast<uint32_t*>(&h);
    }
};
template <> struct Pack8<at::Half> {
    static __device__ __forceinline__ uint32_t two(float a, float b) {
        __half2 h = __floats2half2_rn(a, b);
        return *reinterpret_cast<uint32_t*>(&h);
    }
};
template <typename OutT, int NB>
__device__ __forceinline__ void decode8_fixed(const uint8_t* __restrict__ ids, int64_t A, float ts, float m, float (&w)[8]) {
    const int64_t a0 = A & ~(int64_t)3;
    const uint32_t w0 = __ldg(reinterpret_cast<const uint32_t*>(ids + a0));
    const uint32_t w1 = __ldg(reinterpret_cast<const uint32_t*>(ids + a0 + 4));
    const uint32_t w2 = __ldg(reinterpret_cast<const uint32_t*>(ids + a0 + 8));
    const uint32_t sh = (uint32_t)(A & 3) * 8u;
    const uint32_t lo = __funnelshift_r(w0, w1, sh), hi = __funnelshift_r(w1, w2, sh);
    w[0] = fmaf(small_uint_to_float(field_of<NB, 0>(lo, hi)), ts, m);
    w[1] = fmaf(small_uint_to_float(field_of<NB, 1>(lo, hi)), ts, m);
    w[2] = fmaf(small_uint_to_float(field_of<NB, 2>(lo, hi)), ts, m);
    w[3] = fmaf(small_uint_to_float(field_of<NB, 3>(lo, hi)), ts, m);
    w[4] = fmaf(small_uint_to_float(field_of<NB, 4>(lo, hi)), ts, m);
    w[5] = fmaf(small_uint_to_float(field_of<NB, 5>(lo, hi)), ts, m);
    w[6] = fmaf(small_uint_to_float(field_of<NB, 6>(lo, hi)), ts, m);
    w[7] = fmaf(small_uint_to_float(field_of<NB, 7>(lo, hi)), ts, m);
}
template <typename OutT>
__global__ void __launch_bounds__(256) qvb8_decode_all_v3(
    const uint8_t* __restrict__ ids, const float* __restrict__ min_s,
    const uint8_t* __restrict__ row_nbits, const int64_t* __restrict__ row_bit_off,
    OutT* __restrict__ out, const float* __restrict__ tau_s, int P, int Q, int gx) {
    // 1-D grid of gx blocks per row (no 65535 limit on the row count, e.g. LM heads with Q > 65535);
    // one integer division per block, not per element.
    const int q = (int)(blockIdx.x / (unsigned)gx);
    const int g = (int)(blockIdx.x - (unsigned)q * (unsigned)gx) * blockDim.x + threadIdx.x;   // group of 8 columns
    const int c0 = g * 8;
    if (q >= Q || c0 >= P) return;
    const int nb = row_nbits[q];
    const int64_t base_bit = row_bit_off[q];
    const float m = min_s[q], ts = tau_s[q];
    float w[8];
    const int64_t A = (base_bit >> 3) + (int64_t)g * nb;      // byte of field c0 (8 fields = nb bytes)
    // The fixed-width paths need a byte-aligned row (true for every pack with P % 8 == 0, whose rows start at
    // bit 0 and span P*nb bits); a row that is not takes the per-element path, which works at any bit offset.
    switch ((base_bit & 7) == 0 ? nb : 0) {
        case 4: decode8_fixed<OutT, 4>(ids, A, ts, m, w); break;
        case 5: decode8_fixed<OutT, 5>(ids, A, ts, m, w); break;
        case 6: decode8_fixed<OutT, 6>(ids, A, ts, m, w); break;
        case 7: decode8_fixed<OutT, 7>(ids, A, ts, m, w); break;
        case 8: decode8_fixed<OutT, 8>(ids, A, ts, m, w); break;
        default:
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                w[j] = fmaf((float)load_field(ids, base_bit + (int64_t)(c0 + j) * nb, nb), ts, m);
    }
    uint4 v;
    v.x = Pack8<OutT>::two(w[0], w[1]); v.y = Pack8<OutT>::two(w[2], w[3]);
    v.z = Pack8<OutT>::two(w[4], w[5]); v.w = Pack8<OutT>::two(w[6], w[7]);
    *reinterpret_cast<uint4*>(out + (int64_t)q * P + c0) = v;
}

// Byte-aligned rows and 16-byte stores need P % 8 == 0 and a 4-byte-aligned ids base (as the v3 GEMV).
// Otherwise the per-element kernel runs (the only remaining use of it). Verified bit-identical to the
// per-element kernel on every packed linear of Qwen-Image, Z-Image, Wan2.2 T2V/S2V, Sonic and Qwen3.5-9B.
static bool qvb8_decode_v3_ok(const torch::Tensor& ids, int64_t P) {
    return (P % 8) == 0 && (reinterpret_cast<uintptr_t>(ids.data_ptr<uint8_t>()) & 3u) == 0;
}
template <typename OutT>
static void qvb8_decode_launch(const torch::Tensor& ids, const torch::Tensor& min_s, const torch::Tensor& row_nbits,
                               const torch::Tensor& row_bit_off, torch::Tensor& out, const torch::Tensor& tau_s,
                               int64_t P, int64_t Q, cudaStream_t stream) {
    const int T = 256;
    const int64_t gx = (P / 8 + T - 1) / T;
    if (qvb8_decode_v3_ok(ids, P) && Q > 0 && gx * Q < (int64_t)0x7fffffff) {
        qvb8_decode_all_v3<OutT><<<(unsigned)(gx * Q), T, 0, stream>>>(
            ids.data_ptr<uint8_t>(), min_s.data_ptr<float>(), row_nbits.data_ptr<uint8_t>(),
            row_bit_off.data_ptr<int64_t>(), out.data_ptr<OutT>(), tau_s.data_ptr<float>(), (int)P, (int)Q, (int)gx);
    } else if (Q > 0 && P > 0) {
        const int64_t n = Q * P; const int blocks = (int)((n + T - 1) / T);
        qvb8_decode_all<OutT><<<blocks, T, 0, stream>>>(
            ids.data_ptr<uint8_t>(), min_s.data_ptr<float>(), row_nbits.data_ptr<uint8_t>(),
            row_bit_off.data_ptr<int64_t>(), out.data_ptr<OutT>(), tau_s.data_ptr<float>(), (int)P, (int)Q);
    }
}

torch::Tensor qvb8_decode(torch::Tensor ids, torch::Tensor min_s,
                          torch::Tensor row_nbits, torch::Tensor row_bit_off,
                          int64_t P, torch::Tensor tau_s,
                          torch::Tensor esc_idx, torch::Tensor esc_vals) {
    // Launch on the card the operands live on, not whichever is current:
    // a wrong current device reads another GPU's memory (Xid 31).
    const c10::cuda::OptionalCUDAGuard _qv_guard(device_of(ids));
    const int64_t Q = min_s.numel();
    // The decode dtype is fp16, unconditionally, and that is not a MOCK.
    //
    // This used to be `const bool need_fp32 = false;` with a comment saying
    // the real build takes it from a pack-time flag ("fp16 half-ULP at
    // max|w| > tau"). No such flag was ever added, so the constant stood in
    // for one, and the two branches below were dead code -- a kernel that
    // claimed to handle a case it could never be asked about.
    //
    // The flag is unnecessary because the precondition it would guard is
    // already ENFORCED at pack time, not merely assumed. fp16 cannot represent
    // a value with error below tau once the value is large enough that fp16's
    // half-ULP exceeds it (at tau=1e-3, roughly |w| >= 4), and the packer
    // deals with that by refusing the value into the esc32 pool instead:
    // pack_qvb8._escape_representable raises on exactly that condition, and
    // _calibrate_block routes such positions to escapes. So a pack that exists
    // at all is a pack whose escapes are all fp16-holdable, which is exactly
    // what the certificate measures
    // (|original_precision(w) - restore_precision(calc_fp32)| <= tau, with
    // restore_precision = fp16) -- verified bit-exact against this kernel's
    // (at::Half) cast in certified_error.
    //
    // So decoding to fp32 would not be "more accurate" here, it would be
    // decoding to a PRECISION THE CERTIFICATE NEVER MEASURED, while doubling
    // the buffer and halving the bandwidth on the one path that is supposed to
    // be cheap. If a future pack ever needs fp32 decode, it needs the flag AND
    // a certificate that certifies against fp32 -- in that order, not this one
    // alone.
    constexpr auto dt = torch::kHalf;
    auto out = torch::empty({Q, P}, torch::dtype(dt).device(ids.device()));
    qvb8_decode_launch<at::Half>(ids, min_s, row_nbits, row_bit_off, out, tau_s, P, Q, at::cuda::getCurrentCUDAStream());
    if (esc_idx.numel() > 0) {
        // Escaped positions are stored verbatim; write them over the
        // banded value rather than branching in the decode loop. The
        // cast back to fp16 is what keeps this equal to the value the
        // certificate measured -- esc_vals is fp32 on the wire (wider for a
        // fp32 source) and narrowing it here is the same rounding the
        // certified fp16 path applies.
        out.view({-1}).index_copy_(0, esc_idx, esc_vals.to(dt));
    }
    return out;
}

// Largest M still routed through the GEMV. Above it the decode-once +
// cuBLAS GEMM path wins, because the decode's cost is fixed while the
// GEMV re-decodes the weight for every row and its per-thread accumulators
// eat the register file.
//
// Measured on this box (the dev GPU, 48 SM, cc 12.1), GEMV vs decode+GEMM in ms:
//         M=4    M=8    M=16   M=32   M=300
//  gate   0.59   0.95   1.42   2.66   59.60      GEMM ~2.3 flat
//  down   0.66   1.88   2.91   4.05
//  qkv    0.32   0.49   0.94   1.69
//  head   8.90  13.42  19.36  36.85
// At M=8 all four shapes still win (down_proj by 1.2x, the rest 2.3-3.0x);
// at M=16 down_proj already loses. Hence 8.
//
// THIS IS HARDWARE-SPECIFIC. The crossover is set by register file size,
// SM count and the bandwidth/tensor-core ratio, so a different GPU (V100's
// 80 SM and 900 GB/s HBM2, say) will sit elsewhere. QVB8_GEMV_MAX_M
// overrides it without a rebuild; re-measure before trusting the default
// on new hardware.
static int qvb8_gemv_max_m() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("QVB8_GEMV_MAX_M");
        int v = e ? std::atoi(e) : 8;
        cached = (v >= 1 && v <= 8) ? v : 8;   // only <=8 is instantiated
    }
    return cached;
}


// fp32 -> fp16 that SATURATES instead of overflowing to inf.
//
// The ACTIVATION side needs the range guard the weight side gets from the
// codec, and fp16 has no overflow margin: a Z-Image
// layers.14.feed_forward.w2 input measured absmax 65920 (> fp16's 65504),
// so a plain narrowing turned that whole token row into +inf -- 3840 of
// 31703040 outputs non-finite, which the VAE decode turned into NaN and the
// final uint8 cast into an all-black 1024x1024 image.
//
// Saturating costs nothing: it is two ALU ops inside the cast that was
// already happening, i.e. the same memory traffic as `.to(torch::kHalf)`,
// and every value that FITS fp16 is bit-identical to what the plain cast
// produced. The alternative -- an exact power-of-two rescale with the scale
// undone on the output -- is exact for the out-of-range values too, but
// measurably too expensive here: a reduction plus four extra passes over
// X/Y measured +16% wall clock on the Z-Image 1024x1024 run (5.01s vs
// 4.29s per step), on a pipeline that is memory-bandwidth bound.
//
// Written as comparisons rather than fminf/fmaxf on purpose: CUDA's
// fmaxf/fminf drop a NaN operand, which would silently CONVERT an upstream
// NaN into +-65504. This guard exists to stop INF saturation; NaNs must
// keep propagating so the nan probe still sees them.
//
// What saturating gives up: the error it adds is the OVERSHOOT, and only
// for the elements that do not fit. For the measured case that is 0.6% of
// one activation in a 10240-long dot product (~1e-4 of that output); an
// activation an order of magnitude out of range would be a wrong answer
// instead of the right one -- but it is inf under the plain cast, which
// destroys the whole row and then the image, and nothing measured on this
// box (diffusion or LLM) has come close to that.
__device__ __forceinline__ half sat_half(float v) {
    return __float2half_rn(v > 65504.0f ? 65504.0f
                          : v < -65504.0f ? -65504.0f : v);
}

__global__ void cast_sat_half(const float* __restrict__ x,
                              half* __restrict__ out, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sat_half(x[i]);
    }
}

// bf16 activations straight to saturated fp16, in ONE pass. The model runs in
// bf16, so every linear used to convert bf16 -> fp32 (a full-size copy) and
// then fp32 -> fp16 here; the bf16 -> fp32 step is exact, so reading bf16
// directly gives a bit-identical result for a third of the traffic.
__global__ void cast_sat_half_bf16(const __nv_bfloat16* __restrict__ x,
                                   half* __restrict__ out, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sat_half(__bfloat162float(x[i]));
    }
}

// The activation tensor as fp16, narrowed by cast_sat_half (see above).
// Every call site used to write `.to(torch::kHalf)` here, which is the same
// traffic; this is the single place that turns an activation into the fp16
// the kernels consume, whichever M regime is about to run.
static torch::Tensor saturating_half(const torch::Tensor& X) {
    // Already fp16: the caller needs fp16 operands and fp16 IS the target, so
    // the conversion is the identity. Doing it anyway launches a full-size
    // elementwise kernel to copy every byte through float and back for no
    // change in value or range -- fp16 cannot be over/under-represented by
    // going through fp32.
    //
    // Measured with nsys on the M>8 prefill path (Sonic's biggest linear,
    // Q=1280, P=23040, M=2048), four kernels ran per forward:
    //     nvjet_..._TNNN  (the GEMM)      1.96 ms   needed
    //     qvb8_decode_all                 0.53 ms   needed
    //     cast_sat_half                   1.24 ms   pure waste
    //     direct_copy_kernel              1.44 ms   pure waste
    // 2.7 ms of 5.2 ms was spent re-copying activations that did not change
    // dtype. This early return removes both, on the strength of the caller's
    // dtype, so no cast kernel is launched at all.
    //
    // bf16 and fp32 still convert: the kernels read at::Half, and bf16's wider
    // exponent range must be narrowed deliberately (sat_half, not a plain
    // cast, is what keeps a 65920-valued activation from becoming inf).
    if (X.scalar_type() == torch::kHalf) {
        return X.contiguous();
    }
    if (X.scalar_type() == torch::kBFloat16) {
        auto Xb = X.contiguous();
        auto outb = torch::empty(Xb.sizes(),
                                 torch::dtype(torch::kHalf).device(Xb.device()));
        const int64_t nb = Xb.numel();
        if (nb > 0) {
            const int T = 256;
            cast_sat_half_bf16<<<(int)((nb + T - 1) / T), T, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(Xb.data_ptr<at::BFloat16>()),
                (half*)outb.data_ptr<at::Half>(), nb);
        }
        return outb;
    }
    auto Xf = X.contiguous();
    if (Xf.scalar_type() != torch::kFloat) {
        Xf = Xf.to(torch::kFloat);
    }
    auto out = torch::empty(Xf.sizes(),
                            torch::dtype(torch::kHalf).device(Xf.device()));
    const int64_t n = Xf.numel();
    if (n > 0) {
        const int T = 256;
        cast_sat_half<<<(int)((n + T - 1) / T), T, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
            Xf.data_ptr<float>(), (half*)out.data_ptr<at::Half>(), n);
    }
    return out;
}


// ---------------------------------------------------------------------------
// Fused MoE expert stack (decode: a handful of tokens, top-k experts each).
//
// The per-expert Python loop issued ~12 small kernels per selected expert
// (index_select x3, two GEMVs, activation, two multiplies, index_add_) plus a
// host sync per layer: ~9,000 launches per decode step on the 35B, which left
// the GPU idle a third of the time and spent more time on launches than on
// reading weights. Here every (token, selected expert) PAIR is one row of a
// grouped GEMV, and a layer is five launches: cast, gate_up, activation, down,
// combine.
//
// Requires the stacked layout the Python side builds (see QVBandExpertsV8.
// build_stack): every expert's packed ids concatenated into ONE buffer, with
// per-row metadata laid out (expert, row) and bit offsets made global. Experts
// must carry no escape entries (the stack refuses otherwise). Fixed-width paths
// cover rows of 1..8 bits; wider rows take the generic loop.
template <int BW>
__global__ void __launch_bounds__(BW*WARP) qvb8_moe_gemv(
    const float*   __restrict__ X, int x_by_pair, int topk,
    const int64_t* __restrict__ pair_expert,
    const uint8_t* __restrict__ ids,
    const float*   __restrict__ min_s_all,
    const float*   __restrict__ tau_s_all,
    const uint8_t* __restrict__ nbits_all,
    const int64_t* __restrict__ bit_off_all,
    float*         __restrict__ Y,           // (npairs, Q)
    int P, int Q) {

    const int p = blockIdx.y;
    const int e = (int)pair_expert[p];
    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int q = blockIdx.x * BW + warp;
    if (q >= Q) return;

    const float* Xr = X + (size_t)(x_by_pair ? p : p / topk) * P;
    const size_t mi = (size_t)e * Q + q;
    const int nb = nbits_all[mi];
    const int64_t base_bit = bit_off_all[mi];
    const float m = min_s_all[mi];
    const float ts = tau_s_all[mi];
    float acc[1] = {0.f};
    const int n_blk = (P + 255) / 256;
    const int64_t bb = base_bit >> 3;
    switch (nb) {
        case 1: row_v3_fixed<1, 1>(Xr, ids, bb, P, 1, m, ts, acc, lane, 0, n_blk); break;
        case 2: row_v3_fixed<1, 2>(Xr, ids, bb, P, 1, m, ts, acc, lane, 0, n_blk); break;
        case 3: row_v3_fixed<1, 3>(Xr, ids, bb, P, 1, m, ts, acc, lane, 0, n_blk); break;
        case 4: row_v3_fixed<1, 4>(Xr, ids, bb, P, 1, m, ts, acc, lane, 0, n_blk); break;
        case 5: row_v3_fixed<1, 5>(Xr, ids, bb, P, 1, m, ts, acc, lane, 0, n_blk); break;
        case 6: row_v3_fixed<1, 6>(Xr, ids, bb, P, 1, m, ts, acc, lane, 0, n_blk); break;
        case 7: row_v3_fixed<1, 7>(Xr, ids, bb, P, 1, m, ts, acc, lane, 0, n_blk); break;
        case 8: row_v3_fixed<1, 8>(Xr, ids, bb, P, 1, m, ts, acc, lane, 0, n_blk); break;
        default: {
            for (int c = lane; c < P; c += WARP) {
                const uint32_t id = load_field(ids, base_bit + (int64_t)c * nb, nb);
                acc[0] = fmaf(Xr[c], fmaf((float)id, ts, m), acc[0]);
            }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[0] += __shfl_down_sync(0xffffffff, acc[0], o);
    if (lane == 0) Y[(size_t)p * Q + q] = acc[0];
}

// H[p, i] = silu(Y[p, i]) * Y[p, I + i]   (gate = first half, up = second)
__global__ void qvb8_moe_act(const float* __restrict__ Y, float* __restrict__ H,
                             int I, int64_t n) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int64_t p = idx / I, i = idx % I;
    const float g = Y[p * 2 * (int64_t)I + i];
    const float u = Y[p * 2 * (int64_t)I + I + i];
    H[idx] = (g / (1.0f + __expf(-g))) * u;
}

__device__ __forceinline__ void moe_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void moe_store(__half* p, float v) { *p = __float2half_rn(v); }
__device__ __forceinline__ void moe_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

template <typename OutT>
__global__ void qvb8_moe_combine(const float* __restrict__ Yd,
                                 const float* __restrict__ w,
                                 OutT* __restrict__ out, int topk, int D, int M) {
    const int t = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= M || d >= D) return;
    float a = 0.f;
    for (int j = 0; j < topk; ++j)
        a = fmaf(w[t * topk + j], Yd[((size_t)t * topk + j) * D + d], a);
    moe_store(out + (size_t)t * D + d, a);
}

torch::Tensor qvb8_moe_experts_forward(
    torch::Tensor X, torch::Tensor idx, torch::Tensor w,
    torch::Tensor gu_ids, torch::Tensor gu_min, torch::Tensor gu_tau,
    torch::Tensor gu_nb, torch::Tensor gu_off, int64_t Qgu, int64_t Pgu,
    torch::Tensor dn_ids, torch::Tensor dn_min, torch::Tensor dn_tau,
    torch::Tensor dn_nb, torch::Tensor dn_off, int64_t Qdn, int64_t Pdn) {
    // Launch on the card the operands live on, not whichever is current:
    // a wrong current device reads another GPU's memory (Xid 31).
    const c10::cuda::OptionalCUDAGuard _qv_guard(device_of(X));
    TORCH_CHECK(X.is_cuda() && X.dim() == 2 && X.size(1) == Pgu, "moe: bad X");
    TORCH_CHECK(Qgu % 2 == 0 && Qgu / 2 == Pdn, "moe: gate_up/down shapes disagree");
    TORCH_CHECK(Pgu % 8 == 0 && Pdn % 8 == 0, "moe: P must be a multiple of 8");
    TORCH_CHECK(idx.scalar_type() == torch::kLong && w.scalar_type() == torch::kFloat,
                "moe: idx must be int64 and w float32");
    const int64_t M = X.size(0);
    const int topk = (int)idx.size(1);
    const int64_t npairs = M * topk;
    auto idxc = idx.contiguous();
    auto wc = w.contiguous();
    auto Xf = saturating_half(X).to(torch::kFloat).contiguous();
    auto fopt = torch::dtype(torch::kFloat).device(X.device());
    auto Ygu = torch::empty({npairs, Qgu}, fopt);
    auto H = torch::empty({npairs, Pdn}, fopt);
    auto Yd = torch::empty({npairs, Qdn}, fopt);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int BW = 8;
    qvb8_moe_gemv<BW><<<dim3((Qgu + BW - 1) / BW, (unsigned)npairs), BW * WARP, 0, stream>>>(
        Xf.data_ptr<float>(), 0, topk, idxc.data_ptr<int64_t>(),
        gu_ids.data_ptr<uint8_t>(), gu_min.data_ptr<float>(), gu_tau.data_ptr<float>(),
        gu_nb.data_ptr<uint8_t>(), gu_off.data_ptr<int64_t>(), Ygu.data_ptr<float>(),
        (int)Pgu, (int)Qgu);
    const int64_t nh = npairs * Pdn;
    qvb8_moe_act<<<(int)((nh + 255) / 256), 256, 0, stream>>>(
        Ygu.data_ptr<float>(), H.data_ptr<float>(), (int)Pdn, nh);
    qvb8_moe_gemv<BW><<<dim3((Qdn + BW - 1) / BW, (unsigned)npairs), BW * WARP, 0, stream>>>(
        H.data_ptr<float>(), 1, topk, idxc.data_ptr<int64_t>(),
        dn_ids.data_ptr<uint8_t>(), dn_min.data_ptr<float>(), dn_tau.data_ptr<float>(),
        dn_nb.data_ptr<uint8_t>(), dn_off.data_ptr<int64_t>(), Yd.data_ptr<float>(),
        (int)Pdn, (int)Qdn);
    auto out = torch::empty({M, Qdn}, X.options());
    dim3 g((unsigned)((Qdn + 255) / 256), (unsigned)M);
    if (X.scalar_type() == torch::kBFloat16)
        qvb8_moe_combine<__nv_bfloat16><<<g, 256, 0, stream>>>(
            Yd.data_ptr<float>(), wc.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), topk, (int)Qdn, (int)M);
    else if (X.scalar_type() == torch::kHalf)
        qvb8_moe_combine<__half><<<g, 256, 0, stream>>>(
            Yd.data_ptr<float>(), wc.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), topk, (int)Qdn, (int)M);
    else
        qvb8_moe_combine<float><<<g, 256, 0, stream>>>(
            Yd.data_ptr<float>(), wc.data_ptr<float>(), out.data_ptr<float>(),
            topk, (int)Qdn, (int)M);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Non-gated expert MLP (Nemotron-H): H = relu(Y)^2, Y = up(x) with Qup == Pdn.
__global__ void qvb8_moe_act_relu2(const float* __restrict__ Y, float* __restrict__ H, int64_t n) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = fmaxf(Y[idx], 0.f);
    H[idx] = v * v;
}

torch::Tensor qvb8_moe_experts_relu2_forward(
    torch::Tensor X, torch::Tensor idx, torch::Tensor w,
    torch::Tensor up_ids, torch::Tensor up_min, torch::Tensor up_tau,
    torch::Tensor up_nb, torch::Tensor up_off, int64_t Qup, int64_t Pup,
    torch::Tensor dn_ids, torch::Tensor dn_min, torch::Tensor dn_tau,
    torch::Tensor dn_nb, torch::Tensor dn_off, int64_t Qdn, int64_t Pdn) {
    // Launch on the card the operands live on, not whichever is current:
    // a wrong current device reads another GPU's memory (Xid 31).
    const c10::cuda::OptionalCUDAGuard _qv_guard(device_of(X));
    TORCH_CHECK(X.is_cuda() && X.dim() == 2 && X.size(1) == Pup, "moe_relu2: bad X");
    TORCH_CHECK(Qup == Pdn, "moe_relu2: up/down shapes disagree");
    TORCH_CHECK(Pup % 8 == 0 && Pdn % 8 == 0, "moe_relu2: P must be a multiple of 8");
    TORCH_CHECK(idx.scalar_type() == torch::kLong && w.scalar_type() == torch::kFloat,
                "moe_relu2: idx must be int64 and w float32");
    const int64_t M = X.size(0);
    const int topk = (int)idx.size(1);
    const int64_t npairs = M * topk;
    auto idxc = idx.contiguous();
    auto wc = w.contiguous();
    auto Xf = saturating_half(X).to(torch::kFloat).contiguous();
    auto fopt = torch::dtype(torch::kFloat).device(X.device());
    auto Yup = torch::empty({npairs, Qup}, fopt);
    auto H = torch::empty({npairs, Pdn}, fopt);
    auto Yd = torch::empty({npairs, Qdn}, fopt);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int BW = 8;
    qvb8_moe_gemv<BW><<<dim3((Qup + BW - 1) / BW, (unsigned)npairs), BW * WARP, 0, stream>>>(
        Xf.data_ptr<float>(), 0, topk, idxc.data_ptr<int64_t>(),
        up_ids.data_ptr<uint8_t>(), up_min.data_ptr<float>(), up_tau.data_ptr<float>(),
        up_nb.data_ptr<uint8_t>(), up_off.data_ptr<int64_t>(), Yup.data_ptr<float>(),
        (int)Pup, (int)Qup);
    const int64_t nh = npairs * Pdn;
    qvb8_moe_act_relu2<<<(int)((nh + 255) / 256), 256, 0, stream>>>(
        Yup.data_ptr<float>(), H.data_ptr<float>(), nh);
    qvb8_moe_gemv<BW><<<dim3((Qdn + BW - 1) / BW, (unsigned)npairs), BW * WARP, 0, stream>>>(
        H.data_ptr<float>(), 1, topk, idxc.data_ptr<int64_t>(),
        dn_ids.data_ptr<uint8_t>(), dn_min.data_ptr<float>(), dn_tau.data_ptr<float>(),
        dn_nb.data_ptr<uint8_t>(), dn_off.data_ptr<int64_t>(), Yd.data_ptr<float>(),
        (int)Pdn, (int)Qdn);
    auto out = torch::empty({M, Qdn}, X.options());
    dim3 g((unsigned)((Qdn + 255) / 256), (unsigned)M);
    if (X.scalar_type() == torch::kBFloat16)
        qvb8_moe_combine<__nv_bfloat16><<<g, 256, 0, stream>>>(
            Yd.data_ptr<float>(), wc.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), topk, (int)Qdn, (int)M);
    else if (X.scalar_type() == torch::kHalf)
        qvb8_moe_combine<__half><<<g, 256, 0, stream>>>(
            Yd.data_ptr<float>(), wc.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), topk, (int)Qdn, (int)M);
    else
        qvb8_moe_combine<float><<<g, 256, 0, stream>>>(
            Yd.data_ptr<float>(), wc.data_ptr<float>(), out.data_ptr<float>(),
            topk, (int)Qdn, (int)M);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor qvb8_gemv_multi_forward(torch::Tensor X, torch::Tensor ids,
                                      torch::Tensor min_s,
                                      torch::Tensor row_nbits,
                                      torch::Tensor row_bit_off, int64_t P,
                                      torch::Tensor tau_s, torch::Tensor esc_idx,
                                      torch::Tensor esc_vals) {
    // Launch on the card the operands live on, not whichever is current:
    // a wrong current device reads another GPU's memory (Xid 31).
    const c10::cuda::OptionalCUDAGuard _qv_guard(device_of(X));
    const int64_t Q = min_s.numel();
    const int64_t M = X.size(0);
    // The kernel indexes X and Y by its MB template parameter, so MB must
    // be the ACTUAL row count. Dispatching every 5..8 to MB=8 (as this
    // once did) made the kernel read X rows [M,8) and write Y rows [M,8)
    // out of bounds; M in {5,6,7} is routine -- the MoE expert stack calls
    // this path once per expert per projection, whenever an expert is
    // routed 5-7 tokens in a prefill. One instantiation per M instead, and
    // a range check for anything outside it: qvb8_forward never sends M>8
    // here (that is qvb8_decode's decode-once + cuBLAS GEMM shape), but
    // this function is also exposed directly as `gemv_multi`.
    TORCH_CHECK(M >= 1 && M <= 8, "qvb8 GEMV-multi: M=", M,
                " is outside the instantiated range 1..8; callers above "
                "this (qvb8_forward) route M>8 to decode+cuBLAS");
    auto Xh = saturating_half(X);
    auto Y = torch::empty({M, Q},
        torch::dtype(torch::kFloat).device(X.device()));
    const int BW = 8;
    int blocks = (int)((Q + BW - 1) / BW);
    const int64_t ids_bytes = ids.numel();
    qvb8_check_ids_aligned(ids);
    const bool v3 = qvb8_use_v3(ids, Xh, P);
    const int S = qvb8_split_for(Q, P);
    torch::Tensor Ypart;
    float* ydst = Y.data_ptr<float>();
    if (S > 1) {
        Ypart = torch::empty({(int64_t)S * M, Q},
            torch::dtype(torch::kFloat).device(X.device()));
        ydst = Ypart.data_ptr<float>();
    }
    // v3 reads the activation as fp32. Values are the fp16-narrowed ones (same
    // numbers the v2 kernels multiply), converted ONCE here instead of once per
    // output row inside the kernel.
    torch::Tensor Xf32;
    if (v3) Xf32 = Xh.to(torch::kFloat).contiguous();
    auto launch = [&](auto kern) {
        dim3 grid(blocks, S);
        kern<<<grid, BW*WARP, 0, at::cuda::getCurrentCUDAStream()>>>(
            (const half*)Xh.data_ptr<at::Half>(), ids.data_ptr<uint8_t>(),
            min_s.data_ptr<float>(), row_nbits.data_ptr<uint8_t>(),
            row_bit_off.data_ptr<int64_t>(), ydst,
            tau_s.data_ptr<float>(), (int)P, (int)Q, (int)M, ids_bytes);
    };
    auto launch3 = [&](auto kern) {
        dim3 grid(blocks, S);
        kern<<<grid, BW*WARP, 0, at::cuda::getCurrentCUDAStream()>>>(
            Xf32.data_ptr<float>(), ids.data_ptr<uint8_t>(),
            min_s.data_ptr<float>(), row_nbits.data_ptr<uint8_t>(),
            row_bit_off.data_ptr<int64_t>(), ydst,
            tau_s.data_ptr<float>(), (int)P, (int)Q, (int)M, ids_bytes);
    };
    if (v3) {
        switch (M) {
            case 1:  launch3(qvb8_gemv_multi_v3<8, 1>); break;
            case 2:  launch3(qvb8_gemv_multi_v3<8, 2>); break;
            case 3:  launch3(qvb8_gemv_multi_v3<8, 3>); break;
            case 4:  launch3(qvb8_gemv_multi_v3<8, 4>); break;
            case 5:  launch3(qvb8_gemv_multi_v3<8, 5>); break;
            case 6:  launch3(qvb8_gemv_multi_v3<8, 6>); break;
            case 7:  launch3(qvb8_gemv_multi_v3<8, 7>); break;
            default: launch3(qvb8_gemv_multi_v3<8, 8>); break;
        }
    } else {
        switch (M) {
            case 1:  launch(qvb8_gemv_multi_v2<8, 1>); break;
            case 2:  launch(qvb8_gemv_multi_v2<8, 2>); break;
            case 3:  launch(qvb8_gemv_multi_v2<8, 3>); break;
            case 4:  launch(qvb8_gemv_multi_v2<8, 4>); break;
            case 5:  launch(qvb8_gemv_multi_v2<8, 5>); break;
            case 6:  launch(qvb8_gemv_multi_v2<8, 6>); break;
            case 7:  launch(qvb8_gemv_multi_v2<8, 7>); break;
            default: launch(qvb8_gemv_multi_v2<8, 8>); break;
        }
    }
    if (S > 1) {
        const int64_t n = M * Q;
        qvb8_splitk_combine<<<(int)((n + 255) / 256), 256, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
            ydst, Y.data_ptr<float>(), n, S);
    }

    if (esc_idx.numel() > 0) {
        // One escape-correction pass per input row, reusing the M=1 kernel's
        // correction so the fix lives in one place.
        const int T = 256;
        const int64_t n_esc = esc_idx.numel();
        for (int64_t i = 0; i < M; ++i) {
            auto Yi = Y.select(0, i);
            qvb8_fix_escapes<<<(int)((n_esc + T - 1) / T), T, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
                (const half*)Xh.select(0, i).contiguous().data_ptr<at::Half>(),
                ids.data_ptr<uint8_t>(), min_s.data_ptr<float>(),
                row_nbits.data_ptr<uint8_t>(), row_bit_off.data_ptr<int64_t>(),
                esc_idx.data_ptr<int64_t>(), esc_vals.data_ptr<float>(),
                Yi.data_ptr<float>(), tau_s.data_ptr<float>(), (int)P, n_esc);
        }
    }
    return Y;
}


// ---------------------------------------------------------------------------
// Register-blocked multi-row GEMV for GPUs without the mma path (sm_70).
// v3 gives each output row its own warp, so every warp re-reads all M
// activation rows from L1 for every 8 weights it decodes: ~32 bytes of X per
// ~0.8 byte of weight, and L1 load throughput -- not DRAM -- bounds it. Here a
// warp owns RB rows of ONE bit width (rows sorted by width into 16-row tiles,
// the same perm/tile_nb the mma path uses) and each X vector it loads feeds RB
// rows. Same fp32 math per output as v3 (fmaf order per row unchanged).
// ---------------------------------------------------------------------------
#ifndef QVB8_RB
#define QVB8_RB 4
#endif

template <int MB, int NB, int RB>
__device__ __forceinline__ void rows_rb_fixed(
    const float* __restrict__ X, const uint8_t* __restrict__ ids,
    const int64_t (&base_byte)[RB], int P, int M, const float (&m)[RB], const float (&ts)[RB],
    float (&acc)[RB][MB], int lane, int blk_begin, int blk_end) {
    constexpr int U = 2;
    for (int b0 = blk_begin; b0 < blk_end; b0 += U) {
        uint32_t w[RB][U][3];
        #pragma unroll
        for (int rr = 0; rr < RB; ++rr) {
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                const int blk = b0 + u;
                const int c0 = blk * 256 + lane * 8;
                const bool valid = blk < blk_end;
                const bool live = valid && c0 < P;
                const int64_t A = base_byte[rr] + ((int64_t)(valid ? blk : b0) * 32 + (live ? lane : 0)) * NB;
                const int64_t a0 = A & ~(int64_t)3;
                w[rr][u][0] = __ldg(reinterpret_cast<const uint32_t*>(ids + a0));
                w[rr][u][1] = __ldg(reinterpret_cast<const uint32_t*>(ids + a0 + 4));
                w[rr][u][2] = __ldg(reinterpret_cast<const uint32_t*>(ids + a0 + 8));
            }
        }
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            const int blk = b0 + u;
            if (blk >= blk_end) break;                          // warp-uniform
            const int c0 = blk * 256 + lane * 8;
            const bool live = c0 < P;
            float wv[RB][8];
            #pragma unroll
            for (int rr = 0; rr < RB; ++rr) {
                const int64_t A = base_byte[rr] + ((int64_t)blk * 32 + (live ? lane : 0)) * NB;
                const uint32_t sh = (uint32_t)(A & 3) * 8u;
                const uint32_t lo = __funnelshift_r(w[rr][u][0], w[rr][u][1], sh);
                const uint32_t hi = __funnelshift_r(w[rr][u][1], w[rr][u][2], sh);
                wv[rr][0] = fmaf(small_uint_to_float(field_of<NB, 0>(lo, hi)), ts[rr], m[rr]);
                wv[rr][1] = fmaf(small_uint_to_float(field_of<NB, 1>(lo, hi)), ts[rr], m[rr]);
                wv[rr][2] = fmaf(small_uint_to_float(field_of<NB, 2>(lo, hi)), ts[rr], m[rr]);
                wv[rr][3] = fmaf(small_uint_to_float(field_of<NB, 3>(lo, hi)), ts[rr], m[rr]);
                wv[rr][4] = fmaf(small_uint_to_float(field_of<NB, 4>(lo, hi)), ts[rr], m[rr]);
                wv[rr][5] = fmaf(small_uint_to_float(field_of<NB, 5>(lo, hi)), ts[rr], m[rr]);
                wv[rr][6] = fmaf(small_uint_to_float(field_of<NB, 6>(lo, hi)), ts[rr], m[rr]);
                wv[rr][7] = fmaf(small_uint_to_float(field_of<NB, 7>(lo, hi)), ts[rr], m[rr]);
            }
            #pragma unroll
            for (int r = 0; r < MB; ++r) {
                float4 xa = make_float4(0.f, 0.f, 0.f, 0.f), xb = xa;
                if (live && r < M) {
                    const float4* xp = reinterpret_cast<const float4*>(X + (size_t)r * P + c0);
                    xa = __ldg(xp);
                    xb = __ldg(xp + 1);
                }
                #pragma unroll
                for (int rr = 0; rr < RB; ++rr) {
                    float a = acc[rr][r];
                    a = fmaf(xa.x, wv[rr][0], a);
                    a = fmaf(xa.y, wv[rr][1], a);
                    a = fmaf(xa.z, wv[rr][2], a);
                    a = fmaf(xa.w, wv[rr][3], a);
                    a = fmaf(xb.x, wv[rr][4], a);
                    a = fmaf(xb.y, wv[rr][5], a);
                    a = fmaf(xb.z, wv[rr][6], a);
                    a = fmaf(xb.w, wv[rr][7], a);
                    acc[rr][r] = a;
                }
            }
        }
    }
}

// One row the way v3 does it (any width): used for mixed tiles and edge rows.
template <int MB>
__device__ __forceinline__ void row_any_v3(
    const float* __restrict__ X, const uint8_t* __restrict__ ids, int nb, int64_t base_bit,
    int P, int M, float m, float ts, float (&acc)[MB], int lane, int b_begin, int b_end) {
    if (nb == 6)      row_v3_fixed<MB, 6>(X, ids, base_bit >> 3, P, M, m, ts, acc, lane, b_begin, b_end);
    else if (nb == 7) row_v3_fixed<MB, 7>(X, ids, base_bit >> 3, P, M, m, ts, acc, lane, b_begin, b_end);
    else if (nb == 4) row_v3_fixed<MB, 4>(X, ids, base_bit >> 3, P, M, m, ts, acc, lane, b_begin, b_end);
    else if (nb == 5) row_v3_fixed<MB, 5>(X, ids, base_bit >> 3, P, M, m, ts, acc, lane, b_begin, b_end);
    else if (nb == 8) row_v3_fixed<MB, 8>(X, ids, base_bit >> 3, P, M, m, ts, acc, lane, b_begin, b_end);
    else {
        const int n_iter = (P + WARP - 1) / WARP;
        const int i_begin = b_begin * 8;
        const int i_end = min(n_iter, b_end * 8);
        for (int i = i_begin; i < i_end; ++i) {
            const int c = i * WARP + lane;
            const int safe_lane = (c < P) ? lane : 0;
            const uint32_t id = load_field(ids, base_bit + (int64_t)(i * WARP + safe_lane) * nb, nb);
            if (c < P) {
                const float w = fmaf((float)id, ts, m);
                #pragma unroll
                for (int r = 0; r < MB; ++r)
                    if (r < M) acc[r] = fmaf(__ldg(X + (size_t)r * P + c), w, acc[r]);
            }
        }
    }
}

template <int MB>
__device__ __forceinline__ void warp_store(float (&acc)[MB], float* Y, int Q, int M, int row, int lane) {
    #pragma unroll
    for (int i = 0; i < MB; ++i) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            acc[i] += __shfl_down_sync(0xffffffff, acc[i], o);
    }
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < MB; ++i)
            if (i < M) Y[(int64_t)i * Q + row] = acc[i];
    }
}

template <int BW, int MB>
__global__ void __launch_bounds__(BW*WARP) qvb8_gemv_rb(
    const float* __restrict__ X, const uint8_t* __restrict__ ids,
    const float* __restrict__ min_s, const uint8_t* __restrict__ row_nbits,
    const int64_t* __restrict__ row_bit_off, float* __restrict__ Y,
    const float* __restrict__ tau_s, const int* __restrict__ perm,
    const int* __restrict__ tile_nb, int P, int Q, int M) {
    constexpr int RB = QVB8_RB;
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int pos0 = (blockIdx.x * BW + warp) * RB;          // sorted position of the warp's first row
    if (pos0 >= Q) return;
    const int n_blk = (P + 255) / 256;
    const int per = (n_blk + (int)gridDim.y - 1) / (int)gridDim.y;
    const int b_begin = (int)blockIdx.y * per;
    const int b_end = min(n_blk, b_begin + per);
    Y += (size_t)blockIdx.y * M * Q;
    const int tnb = tile_nb[pos0 / 16];                       // RB divides 16: one tile
    const bool full = pos0 + RB <= Q;
    if (full && tnb >= 4 && tnb <= 8) {
        int row[RB];
        int64_t base[RB];
        float m[RB], ts[RB];
        float acc[RB][MB];
        #pragma unroll
        for (int rr = 0; rr < RB; ++rr) {
            row[rr] = perm[pos0 + rr];
            base[rr] = row_bit_off[row[rr]] >> 3;
            m[rr] = min_s[row[rr]];
            ts[rr] = tau_s[row[rr]];
            #pragma unroll
            for (int i = 0; i < MB; ++i) acc[rr][i] = 0.f;
        }
        switch (tnb) {
            case 4: rows_rb_fixed<MB, 4, RB>(X, ids, base, P, M, m, ts, acc, lane, b_begin, b_end); break;
            case 5: rows_rb_fixed<MB, 5, RB>(X, ids, base, P, M, m, ts, acc, lane, b_begin, b_end); break;
            case 6: rows_rb_fixed<MB, 6, RB>(X, ids, base, P, M, m, ts, acc, lane, b_begin, b_end); break;
            case 7: rows_rb_fixed<MB, 7, RB>(X, ids, base, P, M, m, ts, acc, lane, b_begin, b_end); break;
            default: rows_rb_fixed<MB, 8, RB>(X, ids, base, P, M, m, ts, acc, lane, b_begin, b_end); break;
        }
        #pragma unroll
        for (int rr = 0; rr < RB; ++rr) warp_store<MB>(acc[rr], Y, Q, M, row[rr], lane);
        return;
    }
    for (int rr = 0; rr < RB && pos0 + rr < Q; ++rr) {       // mixed tile / tail: row by row
        const int q = perm[pos0 + rr];
        float acc[MB];
        #pragma unroll
        for (int i = 0; i < MB; ++i) acc[i] = 0.f;
        row_any_v3<MB>(X, ids, row_nbits[q], row_bit_off[q], P, M, min_s[q], tau_s[q], acc,
                       lane, b_begin, b_end);
        warp_store<MB>(acc, Y, Q, M, q, lane);
    }
}

torch::Tensor qvb8_gemv_rb_forward(torch::Tensor X, torch::Tensor ids,
                                   torch::Tensor min_s, torch::Tensor row_nbits,
                                   torch::Tensor row_bit_off, int64_t P,
                                   torch::Tensor tau_s, torch::Tensor esc_idx,
                                   torch::Tensor esc_vals, torch::Tensor perm,
                                   torch::Tensor tile_nb) {
    const c10::cuda::OptionalCUDAGuard _qv_guard(device_of(X));
    const int64_t Q = min_s.size(0);
    const int64_t M = X.size(0);
    TORCH_CHECK(M >= 1 && M <= 8, "qvb8 gemv_rb: M must be 1..8");
    TORCH_CHECK(perm.numel() == Q && tile_nb.numel() == (Q + 15) / 16, "qvb8 gemv_rb: perm/tile_nb");
    auto Xh = saturating_half(X);
    TORCH_CHECK(qvb8_use_v3(ids, Xh, P), "qvb8 gemv_rb: needs the v3 layout preconditions");
    auto Y = torch::empty({M, Q}, torch::dtype(torch::kFloat).device(X.device()));
    const int BW = 8;
    const int RB = QVB8_RB;
    const int blocks = (int)((Q + BW * RB - 1) / (BW * RB));
    // fewer warps than v3 (RB rows each): split P more to keep the GPU full
    int S = qvb8_split_for((Q + RB - 1) / RB, P);
    torch::Tensor Ypart;
    float* ydst = Y.data_ptr<float>();
    if (S > 1) {
        Ypart = torch::empty({(int64_t)S * M, Q}, torch::dtype(torch::kFloat).device(X.device()));
        ydst = Ypart.data_ptr<float>();
    }
    auto Xf32 = Xh.to(torch::kFloat).contiguous();
    auto launch = [&](auto kern) {
        dim3 grid(blocks, S);
        kern<<<grid, BW * WARP, 0, at::cuda::getCurrentCUDAStream()>>>(
            Xf32.data_ptr<float>(), ids.data_ptr<uint8_t>(), min_s.data_ptr<float>(),
            row_nbits.data_ptr<uint8_t>(), row_bit_off.data_ptr<int64_t>(), ydst,
            tau_s.data_ptr<float>(), perm.data_ptr<int>(), tile_nb.data_ptr<int>(),
            (int)P, (int)Q, (int)M);
    };
    switch (M) {
        case 1: launch(qvb8_gemv_rb<8, 1>); break;
        case 2: launch(qvb8_gemv_rb<8, 2>); break;
        case 3: launch(qvb8_gemv_rb<8, 3>); break;
        case 4: launch(qvb8_gemv_rb<8, 4>); break;
        case 5: launch(qvb8_gemv_rb<8, 5>); break;
        case 6: launch(qvb8_gemv_rb<8, 6>); break;
        case 7: launch(qvb8_gemv_rb<8, 7>); break;
        default: launch(qvb8_gemv_rb<8, 8>); break;
    }
    if (S > 1) {
        const int64_t n = M * Q;
        qvb8_splitk_combine<<<(int)((n + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            ydst, Y.data_ptr<float>(), n, S);
    }
    if (esc_idx.numel() > 0) {
        const int T = 256;
        const int64_t n_esc = esc_idx.numel();
        for (int64_t i = 0; i < M; ++i) {
            auto Yi = Y.select(0, i);
            qvb8_fix_escapes<<<(int)((n_esc + T - 1) / T), T, 0, at::cuda::getCurrentCUDAStream()>>>(
                (const half*)Xh.select(0, i).contiguous().data_ptr<at::Half>(),
                ids.data_ptr<uint8_t>(), min_s.data_ptr<float>(),
                row_nbits.data_ptr<uint8_t>(), row_bit_off.data_ptr<int64_t>(),
                esc_idx.data_ptr<int64_t>(), esc_vals.data_ptr<float>(),
                Yi.data_ptr<float>(), tau_s.data_ptr<float>(), (int)P, n_esc);
        }
    }
    return Y;
}


torch::Tensor qvb8_gemv_forward(torch::Tensor X, torch::Tensor ids,
                                torch::Tensor min_s, torch::Tensor row_nbits,
                                torch::Tensor row_bit_off, int64_t P,
                                torch::Tensor tau_s,
                                torch::Tensor esc_idx, torch::Tensor esc_vals) {
    // Launch on the card the operands live on, not whichever is current:
    // a wrong current device reads another GPU's memory (Xid 31).
    const c10::cuda::OptionalCUDAGuard _qv_guard(device_of(X));
    TORCH_CHECK(X.is_cuda() && ids.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(X.scalar_type() == torch::kHalf, "X must be fp16");
    TORCH_CHECK(ids.scalar_type() == torch::kUInt8, "ids must be uint8");
    TORCH_CHECK(min_s.scalar_type() == torch::kFloat, "min_s must be fp32");
    TORCH_CHECK(row_bit_off.scalar_type() == torch::kLong,
                "row_bit_off must be int64");
    const int64_t Q = min_s.numel();
    TORCH_CHECK(X.numel() == P, "X length must equal P");

    qvb8_check_ids_aligned(ids);
    if (qvb8_use_v3(ids, X, P))
        return qvb8_gemv_multi_forward(X.reshape({1, P}), ids, min_s, row_nbits,
                                       row_bit_off, P, tau_s, esc_idx, esc_vals);
    auto Y = torch::empty({1, Q}, torch::dtype(torch::kFloat).device(X.device()));
    const int BW = 8;
    int blocks = (int)((Q + BW - 1) / BW);
    size_t smem = (size_t)((P + 1) & ~1) * sizeof(half);

    // Above this P, smem-staging X leaves too little of the SM's shared
    // memory budget for enough CONCURRENTLY RESIDENT blocks to hide one
    // warp's decode latency behind another's -- measured on this box
    // (the dev GPU, 100 KB smem/SM, BW=8): a down_proj-shaped tensor
    // (Q=5120, P=17408, only 2 blocks/SM resident at 34 KB/block) ran
    // 2.3x SLOWER than a same-element-count transposed shape
    // (Q=17408, P=5120, 9 KB/block, 9 blocks/SM resident) that stays
    // under this threshold. QVB8_GEMV_SMEM_MAX_P overrides it without a
    // rebuild; re-measure before trusting the default on new hardware
    // (this is the same env-var-override pattern qvb8_gemv_max_m uses
    // for its own hardware-specific crossover).
    static int smem_max_p = -1;
    if (smem_max_p < 0) {
        const char* e = std::getenv("QVB8_GEMV_SMEM_MAX_P");
        smem_max_p = e ? std::atoi(e) : 8192;
        if (smem_max_p <= 0) smem_max_p = 8192;
    }
    if (P <= smem_max_p) {
        // The default 48KB dynamic-smem cap is not enough for some P
        // in this regime (P=8192 needs 16KB, comfortably under it, but
        // a caller with a larger threshold could reach it), so opt in.
        static size_t granted = 0;
        if (smem > 48 * 1024 && smem > granted) {
            cudaError_t e = cudaFuncSetAttribute(
                qvb8_gemv_v2<8, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            TORCH_CHECK(e == cudaSuccess, "qvb8 gemv: smem ", smem,
                        " B exceeds the device limit");
            granted = smem;
        }
        const int S = qvb8_split_for(Q, P);
        torch::Tensor Yp;
        float* yd = Y.data_ptr<float>();
        if (S > 1) { Yp = torch::empty({(int64_t)S, Q}, Y.options()); yd = Yp.data_ptr<float>(); }
        qvb8_gemv_v2<8, true><<<dim3(blocks, S), BW*WARP, smem, at::cuda::getCurrentCUDAStream()>>>(
            (const half*)X.data_ptr<at::Half>(), ids.data_ptr<uint8_t>(),
            min_s.data_ptr<float>(), row_nbits.data_ptr<uint8_t>(),
            row_bit_off.data_ptr<int64_t>(), yd,
            tau_s.data_ptr<float>(), (int)P, (int)Q, ids.numel());
        if (S > 1)
            qvb8_splitk_combine<<<(int)((Q + 255) / 256), 256, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
                yd, Y.data_ptr<float>(), Q, S);
    } else {
        const int S = qvb8_split_for(Q, P);
        torch::Tensor Yp;
        float* yd = Y.data_ptr<float>();
        if (S > 1) { Yp = torch::empty({(int64_t)S, Q}, Y.options()); yd = Yp.data_ptr<float>(); }
        qvb8_gemv_v2<8, false><<<dim3(blocks, S), BW*WARP, 0, at::cuda::getCurrentCUDAStream()>>>(
            (const half*)X.data_ptr<at::Half>(), ids.data_ptr<uint8_t>(),
            min_s.data_ptr<float>(), row_nbits.data_ptr<uint8_t>(),
            row_bit_off.data_ptr<int64_t>(), yd,
            tau_s.data_ptr<float>(), (int)P, (int)Q, ids.numel());
        if (S > 1)
            qvb8_splitk_combine<<<(int)((Q + 255) / 256), 256, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
                yd, Y.data_ptr<float>(), Q, S);
    }

    const int64_t n_esc = esc_idx.numel();
    if (n_esc > 0) {
        const int T = 256;
        qvb8_fix_escapes<<<(int)((n_esc + T - 1) / T), T, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
            (const half*)X.data_ptr<at::Half>(), ids.data_ptr<uint8_t>(),
            min_s.data_ptr<float>(), row_nbits.data_ptr<uint8_t>(),
            row_bit_off.data_ptr<int64_t>(), esc_idx.data_ptr<int64_t>(),
            esc_vals.data_ptr<float>(), Y.data_ptr<float>(),
            tau_s.data_ptr<float>(), (int)P, n_esc);
    }
    return Y;
}

// Single entry point for every M, mirroring the earlier format's qv_anch_forward.
// the earlier format's wrapper learned this the hard way: branching on M in Python and
// doing the large-M case there ran an fp32 GEMM on the FP32 pipes instead
// of the tensor cores (measured 55.6 ms vs 7.6 ms), and because it was a
// separate path from the kernel, fixes to the kernel never reached real
// callers. So callers call THIS for any M and it dispatches.
static torch::Tensor qvb8_forward_impl(torch::Tensor X, torch::Tensor ids,
                           torch::Tensor min_s, torch::Tensor row_nbits,
                           torch::Tensor row_bit_off, int64_t P,
                           torch::Tensor tau_s, torch::Tensor esc_idx,
                           torch::Tensor esc_vals, bool out_half) {
    const int64_t M = X.dim() == 1 ? 1 : X.size(0);
    if (M == 1) {
        // The GEMV kernel reads the activation as fp16; narrowing here --
        // saturating, see sat_half -- keeps every caller from having to
        // know that, the same way the earlier format's wrapper owns its kernel's fp16
        // requirement.
        auto x1 = saturating_half(X.dim() == 1 ? X : X.reshape({-1}));
        return qvb8_gemv_forward(x1, ids, min_s, row_nbits, row_bit_off, P,
                                 tau_s, esc_idx, esc_vals);
    }
    if (M <= qvb8_gemv_max_m()) {
        // Small M: one pass over the ids, each decoded weight feeding M
        // accumulators. The weight never leaves a register, so this path
        // also avoids the fp16 narrowing the decode below applies.
        return qvb8_gemv_multi_forward(X, ids, min_s, row_nbits, row_bit_off,
                                       P, tau_s, esc_idx, esc_vals);
    }
    // M > 8 (prefill): decode the matrix once, then one GEMM. The GEMM's
    // OPERANDS are fp16 (so it runs on the tensor cores exactly as before)
    // but its OUTPUT -- and therefore its accumulator width -- is fp32.
    //
    // Why: torch::matmul on two fp16 tensors returns fp16, and an fp16
    // result saturates at 65504. A Z-Image noise_refiner to_out projection
    // feeds this with activations whose absmax is ~264; the products exceed
    // fp16's range and the WHOLE output row came back +inf (measured: 2 of
    // 8192 token rows, 8192 non-finite values), which then poisons the whole
    // image. cuBLAS already accumulates fp16 GEMMs in fp32 internally --
    // asking for an fp32 C just stops the final store from narrowing, so
    // this is strictly MORE accurate than the path it replaces, at no cost
    // in tensor-core throughput.
    //
    // This does not touch the tau certificate: that bounds the RESTORED
    // WEIGHT VALUE, and W here is still exactly the certified fp16 decode.
    // What changes is only the precision of the matmul's RESULT, which was
    // never in scope (see the note below).
    //
    // SCOPE: the tau certificate covers the RESTORED WEIGHT VALUE only
    // (|fp16(decode) - bf16(original)| <= tau, verified bit-exact against
    // pack_qvb8.py's certified_error -- see that function's docstring).
    // It says nothing about what a downstream matmul then does with that
    // value; that is the matmul's own precision question, out of this
    // codec's scope. The M<=8 GEMV above keeps its fma in an fp32
    // register, which is strictly more accurate than this fp16 decode, so
    // certifying THIS path (the stricter of the two) certifies that one
    // automatically -- no separate bound is needed for it. This is not a
    // "the bound doesn't apply here" case: `W` below IS the certified
    // value, unmodified before the GEMM.
    //
    // The M<=8 GEMV paths above narrow X to fp16 as well and use the same
    // saturating cast (sat_half), so every M regime is covered by the one
    // guard -- it costs them nothing either, since it replaces the cast
    // they were already doing.
    auto W = qvb8_decode(ids, min_s, row_nbits, row_bit_off, P, tau_s,
                         esc_idx, esc_vals);          // (Q, P) fp16, certified
    // Same cuBLAS call shape as qv_anchored_cuda.cu's qv_anch_forward (the
    // the earlier format backend), which is the proven one for this row-major layout:
    // row-major (M,P) @ (P,Q) is issued to column-major cuBLAS as
    // C^T = W^T * X^T, i.e. swap the operands and treat every row-major
    // (r,c) buffer as column-major (c,r).
    //
    // cublasGemmEx directly, NOT at::cuda::blas::gemm<at::Half, float>:
    // that wrapper is declared in ATen/cuda/CUDABlas.h but this torch build
    // does not EXPORT it -- libtorch_cuda.so contains no
    // `_ZN2at4cuda4blas4gemm` symbol at all, only the CUDABlas handle and
    // the batched gels/trsm/getrf entries. An extension that calls it
    // therefore compiles and then dies at import with "undefined symbol:
    // at::cuda::blas::gemm<c10::Half, float>". What cuBLAS is asked for
    // here is what that wrapper would have done: CUDA_R_16F operands
    // (tensor cores, fp32 accumulate) with a CUDA_R_32F result, which is
    // what stops the final store from saturating at 65504.
    //
    // The operands themselves are narrowed by sat_half, NOT by
    // `.to(torch::kHalf)`: fp16 has no overflow margin for the ACTIVATION
    // either, and a Z-Image layers.14.feed_forward.w2 input measured
    // absmax 65920 (> fp16's 65504). The plain cast saturated that whole
    // token row to +inf -- 3840 of 31703040 outputs non-finite, which the
    // VAE decode turned into NaN and the final uint8 cast into an all-black
    // 1024x1024 image. Saturating there costs nothing (two ALU ops inside
    // the cast that was already happening) and is bit-identical for every
    // value that fits fp16.
    auto Xh = saturating_half(X);
    const int64_t Q = min_s.numel();
    // OUTPUT PRECISION, and it defaults to fp32.
    //
    // fp32 is the safe choice and is what this shipped with: the GEMM's
    // operands are fp16 and the accumulate is CUBLAS_COMPUTE_32F either way,
    // so the ONLY thing this decides is the dtype of the stored result -- and
    // fp16 tops out at 65504. A token row whose true sum crosses that came
    // back +inf and NaN-poisoned a whole image (see the M>8 comment above for
    // the Z-Image incident: absmax ~264 on the activations feeding the
    // offending projection). fp32 stores nothing, so it cannot fail that way.
    //
    // It is also not slower. fp16 was tried as the default on the theory that
    // halving the result store (168 MB -> 84 MB at M=4096, Q=10240) would pay
    // for itself, and an early measurement appeared to confirm that: 3.273 ->
    // 1.864 ms per call, 1.76x. That measurement was taken while the GPU was
    // busy with another job. Re-measured on an idle GPU the gap reverses --
    // fp16 output 67.73 ms vs fp32 63.91 ms for the same eight large linears,
    // i.e. fp32 FASTER by ~6%, which is the opposite sign. The bandwidth
    // argument is also weaker than it looks: this path is not store-bound, the
    // decode plus GEMM dominates, so shrinking the output does not show up as
    // a proportional saving.
    //
    // So fp16 output bought no speed and kept a live failure mode. QVB8_GEMM_OUT
    // remains as an escape hatch for anyone who measures a different shape and
    // finds otherwise -- set it to "fp16" to get the narrow store back.
    const char* out_env = std::getenv("QVB8_GEMM_OUT");
    // `out_half` is the per-call form of the same switch, for callers whose
    // own dtype is fp16 (a dense fp16 model already produces fp16 here, so the
    // fp32 store buys no range and costs 2x the result traffic).
    const bool out_fp16 = out_half || (out_env && std::strcmp(out_env, "fp16") == 0);
    auto Y = torch::empty({M, Q},
                          torch::dtype(out_fp16 ? torch::kHalf : torch::kFloat)
                              .device(X.device()));
    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,            // W is (Q,P) row-major = (P,Q) col-major; want W^T
        CUBLAS_OP_N,            // X is (M,P) row-major = (P,M) col-major
        (int)Q, (int)M, (int)P,
        &alpha,
        W.data_ptr<at::Half>(), CUDA_R_16F, (int)P,   // lda = P (row stride of W)
        Xh.data_ptr<at::Half>(), CUDA_R_16F, (int)P,  // ldb = P (row stride of X)
        &beta,
        out_fp16 ? (void*)Y.data_ptr<at::Half>() : Y.data_ptr<float>(),
        out_fp16 ? CUDA_R_16F : CUDA_R_32F, (int)Q,  // ldc = Q (row stride of Y)
        CUBLAS_COMPUTE_32F,
        // CUBLAS_GEMM_DEFAULT, not CUBLAS_GEMM_DEFAULT_TENSOR_OP.
        //
        // DEFAULT_TENSOR_OP is documented as picking an algorithm WITHOUT
        // running cuBLAS's heuristic query -- cuBLAS then guesses from the
        // operand descriptors alone, and the guess it makes for this shape
        // (M=2048, N=1280, K=23040, fp16 operands, fp32 C) is materially
        // worse than what the query returns. torch's own matmul on the same
        // shape and the same layouts runs in ~1.9 ms via DEFAULT; this call
        // took 6.1 ms. The components only account for 2.7 ms of that
        // (GEMM 2.0, decode 0.66, the fp32 store 0.04), so ~3.4 ms was
        // cuBLAS choosing a bad algorithm every single call.
        //
        // The heuristic cost is a query, not a plan change: it is cached by
        // the handle across calls of the same shape, which is exactly why a
        // steady-state loop (one decode+prefill per layer, 525 layers, 15
        // steps) amortises it to nothing while a single cold call pays it.
        CUBLAS_GEMM_DEFAULT));
    return Y;
}

torch::Tensor qvb8_forward(torch::Tensor X, torch::Tensor ids,
                           torch::Tensor min_s, torch::Tensor row_nbits,
                           torch::Tensor row_bit_off, int64_t P,
                           torch::Tensor tau_s, torch::Tensor esc_idx,
                           torch::Tensor esc_vals) {
    // Launch on the card the operands live on, not whichever is current:
    // a wrong current device reads another GPU's memory (Xid 31).
    const c10::cuda::OptionalCUDAGuard _qv_guard(device_of(X));
    return qvb8_forward_impl(X, ids, min_s, row_nbits, row_bit_off, P, tau_s,
                             esc_idx, esc_vals, false);
}

// Same as forward, but the M > 8 GEMM stores fp16. M <= 8 paths still return
// fp32 (their kernels accumulate and store in fp32); callers narrow those.
torch::Tensor qvb8_forward_half(torch::Tensor X, torch::Tensor ids,
                                torch::Tensor min_s, torch::Tensor row_nbits,
                                torch::Tensor row_bit_off, int64_t P,
                                torch::Tensor tau_s, torch::Tensor esc_idx,
                                torch::Tensor esc_vals) {
    // Launch on the card the operands live on, not whichever is current:
    // a wrong current device reads another GPU's memory (Xid 31).
    const c10::cuda::OptionalCUDAGuard _qv_guard(device_of(X));
    return qvb8_forward_impl(X, ids, min_s, row_nbits, row_bit_off, P, tau_s,
                             esc_idx, esc_vals, true);
}

// Full dense decode straight to bf16, for callers whose MODEL precision is
// bf16 (diffusion transformers: Qwen-Image, Z-Image). The fp32 band value
// fma(id, tau, min) is rounded ONCE to bf16 (round-to-nearest-even), not via
// fp16, so there is no double rounding.
//
// This is NOT the certified path. The certificate measures
// |orig(w) - fp16(band)| <= tau; bf16 restore adds up to half a bf16 ULP on
// top. Two facts bound that: (1) where tau < half a bf16 ULP of w (|w| >=
// 256*tau for a bf16 source), the nearest bf16 to the band value IS the
// original weight, so large weights come back bit-exact; (2) elsewhere the
// extra error is <= half a bf16 ULP < tau. Opt-in per module
// (QVBandLinearV8.bf16_gemm); every existing caller keeps the fp16 decode.
torch::Tensor qvb8_decode_bf16(torch::Tensor ids, torch::Tensor min_s,
                               torch::Tensor row_nbits, torch::Tensor row_bit_off,
                               int64_t P, torch::Tensor tau_s,
                               torch::Tensor esc_idx, torch::Tensor esc_vals) {
    // Launch on the card the operands live on, not whichever is current:
    // a wrong current device reads another GPU's memory (Xid 31).
    const c10::cuda::OptionalCUDAGuard _qv_guard(device_of(ids));
    const int64_t Q = min_s.numel();
    auto out = torch::empty({Q, P}, torch::dtype(torch::kBFloat16).device(ids.device()));
    qvb8_decode_launch<at::BFloat16>(ids, min_s, row_nbits, row_bit_off, out, tau_s, P, Q, at::cuda::getCurrentCUDAStream());
    if (esc_idx.numel() > 0) {
        // Escapes are stored verbatim (fp32 on the wire); a bf16 source's
        // escape value is exactly representable, so this narrowing is exact.
        out.view({-1}).index_copy_(0, esc_idx, esc_vals.to(torch::kBFloat16));
    }
    return out;
}


// ---------------------------------------------------------------------------
// Tensor-core small-M path (M = 1..16): the speculative-verify shape (7 drafts + 1 per request, so one
// request is 8 rows and two requests 16).
//
// The v3 GEMV spends MB fp32 FMAs on every decoded weight, so verifying 8 tokens costs ~1.5-2x the
// single-token pass even though the packed ids are read once, and above 8 rows the old route decoded
// the whole matrix and ran a GEMM (~3.5x slower than this on the 27B's big matrices at 9-16 rows).
// Here a warp owns a 16-row output tile, decodes its weights to fp16 and feeds mma.m16n8k16 with the
// activations as the n=8 operand (two n=8 halves for 9..16 rows, sharing the decoded weights).
//
// K PERMUTATION. mma's A fragment gives thread (g=lane/4, c=lane%4) the k slots {2c, 2c+1, 2c+8,
// 2c+9}. The sum over k is order-free, so slots are mapped to whatever physical columns suit the
// loads, applied to A and B alike (see "SUPER-TILE" below).
//
// WEIGHT VALUE. fp16(min_s + id*tau_s), rounded once: the certified restore the M>8 decode path
// already used (the GEMV kept the fp32 value). Accumulation is fp32. Output is fp32 (M, Q) like the GEMV.
//
// BF16 PACKS. A bf16-certified pack is certified in bf16, so it runs the same kernel with bf16 operands
// (weights rounded once to bf16 -- what ext.decode_bf16 produces -- and bf16 activations).
//
// ESCAPES. Weights stored exactly because they left their row's band are patched after the main pass
// (qvb8_mma_fix_escapes), so a matrix with escapes is handled like any other.
//
// ELIGIBILITY (checked once per module by QVBandLinearV8._mma_order, and again by the launcher):
// P a multiple of 128, 4-byte-aligned ids, row widths 1..14 bits.
template <bool BF>
__device__ __forceinline__ void mma_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                          uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // m16n8k16 is sm_80+. Pre-Ampere (V100) never launches the mma path: the
    // host side (qvb8_mma_ok / QVBandLinearV8._mma_order) refuses it there.
    __trap();
#else
    if constexpr (BF) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
#endif
}

#ifndef QVB8_MMA_U
#define QVB8_MMA_U 4
#endif
// One warp = one 16-row output tile; gridDim.y splits the K range (partials go
// to Y + y*M*Q, summed by qvb8_splitk_combine).
//
// SUPER-TILE (128 columns). Thread (g, c) owns, per row, the 32 contiguous columns
// [32c, 32c+32) of a super-tile: 4*NB contiguous bytes. The four c-lanes of a row
// together read ONE contiguous 16*NB-byte span (96..128 B, whole cache sectors)
// where a 32-column tile gave four 24..32 B pieces per row per step and left the
// kernel limited by its weight loads (ablating them took the run from 0.63 ms to
// 0.085 ms; decode and mma are cheap). The K-slot <-> column map is the same for A
// and B: in group j (0..3) of the super-tile, thread c's slots {2c,2c+1,2c+8,2c+9}
// are columns 32c + 8j + {0..3} (first mma) and 32c + 8j + {4..7} (second).
//
// ROW ORDER. The warp's tile is 16 consecutive rows of the width-SORTED order
// (perm), so most tiles hold one bit width and every shift is compile-time (NB);
// tile_nb[tile] is that width, or 0 for a mixed tile, which takes the runtime-width
// path. perm == nullptr means natural order, all tiles mixed.
#define QVB8_MMA_U 2
// Words a thread loads per row and super-tile: a fixed width (4..8 bits) needs 4*NB <= 32 B + up to 3 B of
// misalignment + the funnel/window words = 10; the any-width path (rows of 1..14 bits) up to 56 B + 3 + 1 = 18.
// 9..12-bit rows (v1): the thread's 32 fields span 32*NB bits = NB words; the last field may straddle
// into one more, and the re-based stream drops one word, so NB + 3 loaded words.
template <int NB> struct QvbMmaWords { static constexpr int value = NB == 0 ? 18 : (NB <= 8 ? 10 : NB + 3); };

// Field (J*8 + K) of an NB-bit stream (NB 9..12), every index and shift resolved at compile time (registers only).
template <int NB, int J, int K, int NV>
__device__ __forceinline__ uint32_t field_w(const uint32_t (&v)[NV]) {
    constexpr int B = (J * 8 + K) * NB, W = B >> 5, S = B & 31;
    constexpr uint32_t MASK = (1u << NB) - 1u;
    static_assert(W + 1 < NV || S + NB <= 32, "field_w: stream too short");
    if constexpr (S + NB <= 32) return (v[W] >> S) & MASK;
    else return __funnelshift_r(v[W], v[W + 1], S) & MASK;
}

template <int NB, int J, int NV, int NT, bool BF>
__device__ __forceinline__ void qvb8_mma_group(
    float (&acc)[4], float (&acc2)[4], const uint32_t (&va)[NV], const uint32_t (&vb)[NV],
    int nba, int nbb, uint32_t maska, uint32_t maskb,
    float ma, float mb, float tsa, float tsb, const uint4 xq, const uint4 xq2) {
    uint32_t loa = 0, hia = 0, lob = 0, hib = 0;
    if constexpr (NB != 0 && NB <= 8) {
        constexpr int WI = (J * NB) >> 2, B = ((J * NB) & 3) * 8;
        loa = __funnelshift_r(va[WI], va[WI + 1], B);  hia = __funnelshift_r(va[WI + 1], va[WI + 2], B);
        lob = __funnelshift_r(vb[WI], vb[WI + 1], B);  hib = __funnelshift_r(vb[WI + 1], vb[WI + 2], B);
    }
    float fa[8], fb[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint32_t ia, ib;
        if constexpr (NB > 8) {
            switch (k) {
                case 0: ia = field_w<NB, J, 0>(va); ib = field_w<NB, J, 0>(vb); break;
                case 1: ia = field_w<NB, J, 1>(va); ib = field_w<NB, J, 1>(vb); break;
                case 2: ia = field_w<NB, J, 2>(va); ib = field_w<NB, J, 2>(vb); break;
                case 3: ia = field_w<NB, J, 3>(va); ib = field_w<NB, J, 3>(vb); break;
                case 4: ia = field_w<NB, J, 4>(va); ib = field_w<NB, J, 4>(vb); break;
                case 5: ia = field_w<NB, J, 5>(va); ib = field_w<NB, J, 5>(vb); break;
                case 6: ia = field_w<NB, J, 6>(va); ib = field_w<NB, J, 6>(vb); break;
                default: ia = field_w<NB, J, 7>(va); ib = field_w<NB, J, 7>(vb); break;
            }
        } else if constexpr (NB != 0) {
            switch (k) {
                case 0: ia = field_of<NB, 0>(loa, hia); ib = field_of<NB, 0>(lob, hib); break;
                case 1: ia = field_of<NB, 1>(loa, hia); ib = field_of<NB, 1>(lob, hib); break;
                case 2: ia = field_of<NB, 2>(loa, hia); ib = field_of<NB, 2>(lob, hib); break;
                case 3: ia = field_of<NB, 3>(loa, hia); ib = field_of<NB, 3>(lob, hib); break;
                case 4: ia = field_of<NB, 4>(loa, hia); ib = field_of<NB, 4>(lob, hib); break;
                case 5: ia = field_of<NB, 5>(loa, hia); ib = field_of<NB, 5>(lob, hib); break;
                case 6: ia = field_of<NB, 6>(loa, hia); ib = field_of<NB, 6>(lob, hib); break;
                default: ia = field_of<NB, 7>(loa, hia); ib = field_of<NB, 7>(lob, hib); break;
            }
        } else {
            // Any width (1..14 bits), runtime: field (J*8+k) of this thread's stream sits at bit
            // (J*8+k)*nb, inside two adjacent words. Dynamic register index -> local memory, but
            // only mixed-width and outlier-width tiles (a few per matrix) come here.
            const int bita = (J * 8 + k) * nba, bitb = (J * 8 + k) * nbb;
            ia = __funnelshift_r(va[bita >> 5], va[(bita >> 5) + 1], bita & 31) & maska;
            ib = __funnelshift_r(vb[bitb >> 5], vb[(bitb >> 5) + 1], bitb & 31) & maskb;
        }
        fa[k] = fmaf(small_uint_to_float(ia), tsa, ma);
        fb[k] = fmaf(small_uint_to_float(ib), tsb, mb);
    }
    // The restored weight, rounded once: to fp16 (fp16 packs) or to bf16 (bf16-certified packs, whose
    // certificate is |orig - bf16(band)| <= tau -- the same value ext.decode_bf16 produces).
    auto h2 = [](float lo, float hi) {
        if constexpr (BF) {
            __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
            return *reinterpret_cast<uint32_t*>(&v);
        } else {
            __half2 v = __floats2half2_rn(lo, hi);
            return *reinterpret_cast<uint32_t*>(&v);
        }
    };
    const uint32_t a0 = h2(fa[0], fa[1]), a1 = h2(fb[0], fb[1]), a2 = h2(fa[2], fa[3]), a3 = h2(fb[2], fb[3]);
    const uint32_t a4 = h2(fa[4], fa[5]), a5 = h2(fb[4], fb[5]), a6 = h2(fa[6], fa[7]), a7 = h2(fb[6], fb[7]);
    mma_16816<BF>(acc, a0, a1, a2, a3, xq.x, xq.y);
    mma_16816<BF>(acc, a4, a5, a6, a7, xq.z, xq.w);
    if constexpr (NT == 2) {                         // tokens 8..15 reuse the decoded weights
        mma_16816<BF>(acc2, a0, a1, a2, a3, xq2.x, xq2.y);
        mma_16816<BF>(acc2, a4, a5, a6, a7, xq2.z, xq2.w);
    }
}

template <int NB, int NT, bool BF>
__device__ __forceinline__ void qvb8_mma_tile(
    const half* __restrict__ X, const uint8_t* __restrict__ ids,
    float* __restrict__ Y, int P, int Q, int M, int g, int c, bool va, bool vb,
    int ra, int rb, int nba_rt, int nbb_rt, int64_t baseA, int64_t baseB,
    float ma, float mb, float tsa, float tsb, int s_begin, int s_end, int64_t ids_bytes) {
    const int nba = NB ? NB : nba_rt, nbb = NB ? NB : nbb_rt;
    const uint32_t maska = (1u << nba) - 1u, maskb = (1u << nbb) - 1u;
    constexpr int NW = QvbMmaWords<NB>::value;
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    // A thread reads NW words from its (aligned-down) start, more than a
    // short row field needs; for the physically last row that can pass the end of ids.
    // The words that matter end within the packer's 8 trailing pad bytes, so clamping
    // the rest changes only bits no field reads.
    const int64_t last_word = ids_bytes - 4;
    const half* xrow = X + (size_t)g * P + c * 32;
    const bool xlive = g < M;
    const half* xrow2 = X + (size_t)(g + 8) * P + c * 32;          // tokens 8..15 (NT == 2)
    const bool xlive2 = NT == 2 && (g + 8) < M;
    for (int st0 = s_begin; st0 < s_end; st0 += QVB8_MMA_U) {
        uint32_t wa[QVB8_MMA_U][NW], wb[QVB8_MMA_U][NW];
        #pragma unroll
        for (int u = 0; u < QVB8_MMA_U; ++u) {
            const int st = min(st0 + u, s_end - 1);
            const int64_t a0 = (baseA + (int64_t)(st * 16 + c * 4) * nba) & ~(int64_t)3;
            const int64_t b0 = (baseB + (int64_t)(st * 16 + c * 4) * nbb) & ~(int64_t)3;
            #pragma unroll
            for (int i = 0; i < NW; ++i) {
                wa[u][i] = __ldg(reinterpret_cast<const uint32_t*>(ids + min(a0 + 4 * i, last_word)));
                wb[u][i] = __ldg(reinterpret_cast<const uint32_t*>(ids + min(b0 + 4 * i, last_word)));
            }
        }
        #pragma unroll
        for (int u = 0; u < QVB8_MMA_U; ++u) {
            const int st = st0 + u;
            if (st >= s_end) break;                                // warp-uniform
            const uint32_t sha = (uint32_t)((baseA + (int64_t)(st * 16 + c * 4) * nba) & 3) * 8u;
            const uint32_t shb = (uint32_t)((baseB + (int64_t)(st * 16 + c * 4) * nbb) & 3) * 8u;
            uint32_t vA[NW - 1], vB[NW - 1];   // the stream re-based to start at this thread's first byte
            #pragma unroll
            for (int i = 0; i < NW - 1; ++i) {
                vA[i] = __funnelshift_r(wa[u][i], wa[u][i + 1], sha);
                vB[i] = __funnelshift_r(wb[u][i], wb[u][i + 1], shb);
            }
            const uint4* xp = reinterpret_cast<const uint4*>(xrow + (size_t)st * 128);
            uint4 x0 = make_uint4(0u, 0u, 0u, 0u), x1 = x0, x2 = x0, x3 = x0;
            if (xlive) { x0 = __ldg(xp); x1 = __ldg(xp + 1); x2 = __ldg(xp + 2); x3 = __ldg(xp + 3); }
            uint4 y0 = make_uint4(0u, 0u, 0u, 0u), y1 = y0, y2 = y0, y3 = y0;
            if (xlive2) {
                const uint4* yp = reinterpret_cast<const uint4*>(xrow2 + (size_t)st * 128);
                y0 = __ldg(yp); y1 = __ldg(yp + 1); y2 = __ldg(yp + 2); y3 = __ldg(yp + 3);
            }
            qvb8_mma_group<NB, 0, NW - 1, NT, BF>(acc, acc2, vA, vB, nba, nbb, maska, maskb, ma, mb, tsa, tsb, x0, y0);
            qvb8_mma_group<NB, 1, NW - 1, NT, BF>(acc, acc2, vA, vB, nba, nbb, maska, maskb, ma, mb, tsa, tsb, x1, y1);
            qvb8_mma_group<NB, 2, NW - 1, NT, BF>(acc, acc2, vA, vB, nba, nbb, maska, maskb, ma, mb, tsa, tsb, x2, y2);
            qvb8_mma_group<NB, 3, NW - 1, NT, BF>(acc, acc2, vA, vB, nba, nbb, maska, maskb, ma, mb, tsa, tsb, x3, y3);
        }
    }
    // C fragment: acc[0..1] = row g,   tokens 2c, 2c+1;  acc[2..3] = row g+8, same tokens.
    const int n0 = 2 * c, n1 = 2 * c + 1;
    if (va) {
        if (n0 < M) Y[(int64_t)n0 * Q + ra] = acc[0];
        if (n1 < M) Y[(int64_t)n1 * Q + ra] = acc[1];
    }
    if (vb) {
        if (n0 < M) Y[(int64_t)n0 * Q + rb] = acc[2];
        if (n1 < M) Y[(int64_t)n1 * Q + rb] = acc[3];
    }
    if constexpr (NT == 2) {
        if (va) {
            if (n0 + 8 < M) Y[(int64_t)(n0 + 8) * Q + ra] = acc2[0];
            if (n1 + 8 < M) Y[(int64_t)(n1 + 8) * Q + ra] = acc2[1];
        }
        if (vb) {
            if (n0 + 8 < M) Y[(int64_t)(n0 + 8) * Q + rb] = acc2[2];
            if (n1 + 8 < M) Y[(int64_t)(n1 + 8) * Q + rb] = acc2[3];
        }
    }
}

// v1: shared-memory staged variant of qvb8_mma_tile for fixed-width tiles (NB 4..8).
// The register version has each lane load its own row's words with 4-byte __ldg: one load instruction then
// addresses 8 rows x 4 lanes at a 4*NB-byte stride, i.e. ~32 cache sectors for 128 useful bytes, every
// instruction. Here the warp first copies each of its 16 rows' super-tile spans into shared memory with
// consecutive lanes on consecutive words (one or two sectors per row per instruction), then every lane
// reads its window from shared memory. The decode, the mma and the output are the register version's.
#ifndef QVB8_MMA_SMEM
#define QVB8_MMA_SMEM 1
#endif
constexpr int QVB8_SM_RW = 56;                       // words per staged row: ((3 + 12*NB) >> 2) + NW + 1 <= 55 for NB <= 12
#ifndef QVB8_SM_TPF
#define QVB8_SM_TPF 1
#endif
// staged row stride of the <=16-token smem tile: TPF super-tiles at NB=8
constexpr int QVB8_SM_RWT = ((3 + (QVB8_SM_TPF - 1) * 16 * 12 + 12 * 12) >> 2) + (12 + 3) + 1;   // NB <= 12
template <int NB, int NT, bool BF>
__device__ __forceinline__ void qvb8_mma_tile_smem(
    const half* __restrict__ X, const uint8_t* __restrict__ ids,
    float* __restrict__ Y, int P, int Q, int M, int g, int c, int lane, bool va, bool vb,
    int ra, int rb, int64_t baseA, int64_t baseB,
    float ma, float mb, float tsa, float tsb, int s_begin, int s_end, int64_t ids_bytes,
    uint32_t* __restrict__ buf, int64_t* __restrict__ sb) {
    // QVB8_SM_TPF consecutive super-tiles per fill: each row's staged span is TPF*16*NB contiguous bytes
    // (the kernel is bound by its load pattern -- decode and mma are hidden -- and longer runs per row
    // stream better than 16 scattered ~112-byte pieces).
    constexpr int TPF = QVB8_SM_TPF;
    constexpr int NW = QvbMmaWords<NB>::value;
    constexpr int RWN = ((3 + (TPF - 1) * 16 * NB + 12 * NB) >> 2) + NW + 1;
    static_assert(RWN <= QVB8_SM_RWT, "staged row too wide");
    constexpr int TOTAL = 16 * RWN;
    const uint32_t mask = (1u << NB) - 1u;
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    const int64_t last_word = ids_bytes - 4;
    if (c == 0) { sb[g] = baseA; sb[g + 8] = baseB; }
    __syncwarp();
    const half* xrow = X + (size_t)g * P + c * 32;
    const bool xlive = g < M;
    const half* xrow2 = X + (size_t)(g + 8) * P + c * 32;
    const bool xlive2 = NT == 2 && (g + 8) < M;
    constexpr int NL = (TOTAL + WARP - 1) / WARP;
    uint32_t ld[NL];
    auto issue = [&](int st) {                       // loads of super-tiles st .. st+TPF-1 into registers
        const int64_t sto = (int64_t)st * 16 * NB;
        #pragma unroll
        for (int k = 0; k < NL; ++k) {
            const int idx = lane + k * WARP;
            const int r = min(idx / RWN, 15), w = idx - (idx / RWN) * RWN;
            const int64_t a = ((sb[r] + sto) & ~(int64_t)3) + 4 * w;
            ld[k] = idx < TOTAL ? __ldg(reinterpret_cast<const uint32_t*>(ids + min(a, last_word))) : 0u;
        }
    };
    auto commit = [&](uint32_t* b) {
        #pragma unroll
        for (int k = 0; k < NL; ++k) {
            const int idx = lane + k * WARP;
            if (idx < TOTAL) b[(idx / RWN) * QVB8_SM_RWT + idx - (idx / RWN) * RWN] = ld[k];
        }
    };
    if (s_begin < s_end) { issue(s_begin); commit(buf); }
    __syncwarp();
    for (int st0 = s_begin; st0 < s_end; st0 += TPF) {
        const int f = (st0 - s_begin) / TPF;
        uint32_t* cur = buf + (f & 1) * 16 * QVB8_SM_RWT;
        uint32_t* nxt = buf + ((f & 1) ^ 1) * 16 * QVB8_SM_RWT;
        if (st0 + TPF < s_end) issue(st0 + TPF);      // next fill's loads fly while this one computes
        const int64_t sto0 = (int64_t)st0 * 16 * NB;
        #pragma unroll
        for (int u = 0; u < TPF; ++u) {
            const int st = st0 + u;
            if (st >= s_end) break;                   // warp-uniform
            const int offA = (int)((baseA + sto0) & 3) + u * 16 * NB + c * 4 * NB;
            const int offB = (int)((baseB + sto0) & 3) + u * 16 * NB + c * 4 * NB;
            const uint32_t* pa = cur + g * QVB8_SM_RWT + (offA >> 2);
            const uint32_t* pb = cur + (g + 8) * QVB8_SM_RWT + (offB >> 2);
            const uint32_t sha = (uint32_t)(offA & 3) * 8u, shb = (uint32_t)(offB & 3) * 8u;
            uint32_t vA[NW - 1], vB[NW - 1];
            #pragma unroll
            for (int i = 0; i < NW - 1; ++i) {
                vA[i] = __funnelshift_r(pa[i], pa[i + 1], sha);
                vB[i] = __funnelshift_r(pb[i], pb[i + 1], shb);
            }
            const uint4* xp = reinterpret_cast<const uint4*>(xrow + (size_t)st * 128);
            uint4 x0 = make_uint4(0u, 0u, 0u, 0u), x1 = x0, x2 = x0, x3 = x0;
            if (xlive) { x0 = __ldg(xp); x1 = __ldg(xp + 1); x2 = __ldg(xp + 2); x3 = __ldg(xp + 3); }
            uint4 y0 = make_uint4(0u, 0u, 0u, 0u), y1 = y0, y2 = y0, y3 = y0;
            if (xlive2) {
                const uint4* yp = reinterpret_cast<const uint4*>(xrow2 + (size_t)st * 128);
                y0 = __ldg(yp); y1 = __ldg(yp + 1); y2 = __ldg(yp + 2); y3 = __ldg(yp + 3);
            }
            qvb8_mma_group<NB, 0, NW - 1, NT, BF>(acc, acc2, vA, vB, NB, NB, mask, mask, ma, mb, tsa, tsb, x0, y0);
            qvb8_mma_group<NB, 1, NW - 1, NT, BF>(acc, acc2, vA, vB, NB, NB, mask, mask, ma, mb, tsa, tsb, x1, y1);
            qvb8_mma_group<NB, 2, NW - 1, NT, BF>(acc, acc2, vA, vB, NB, NB, mask, mask, ma, mb, tsa, tsb, x2, y2);
            qvb8_mma_group<NB, 3, NW - 1, NT, BF>(acc, acc2, vA, vB, NB, NB, mask, mask, ma, mb, tsa, tsb, x3, y3);
        }
        if (st0 + TPF < s_end) commit(nxt);
        __syncwarp();
    }
    const int n0 = 2 * c, n1 = 2 * c + 1;
    if (va) {
        if (n0 < M) Y[(int64_t)n0 * Q + ra] = acc[0];
        if (n1 < M) Y[(int64_t)n1 * Q + ra] = acc[1];
    }
    if (vb) {
        if (n0 < M) Y[(int64_t)n0 * Q + rb] = acc[2];
        if (n1 < M) Y[(int64_t)n1 * Q + rb] = acc[3];
    }
    if constexpr (NT == 2) {
        if (va) {
            if (n0 + 8 < M) Y[(int64_t)(n0 + 8) * Q + ra] = acc2[0];
            if (n1 + 8 < M) Y[(int64_t)(n1 + 8) * Q + ra] = acc2[1];
        }
        if (vb) {
            if (n0 + 8 < M) Y[(int64_t)(n0 + 8) * Q + rb] = acc2[2];
            if (n1 + 8 < M) Y[(int64_t)(n1 + 8) * Q + rb] = acc2[3];
        }
    }
}

// ---- 17..48 tokens (NT 3..6): generic token-group versions; <= 16 tokens keep the functions above verbatim.
// NT token groups of 8 (v1: NT up to 6, i.e. up to 48 tokens). Each weight fragment is decoded ONCE and
// fed to NT mma pairs, so a batched verify (several requests x 16 tokens) costs little more than one: the
// decode, not the mma, is the expensive part. The mma order for groups 0 and 1 is the old NT<=2 order, so
// M <= 16 results are bit-identical to before.
template <int NB, int J, int NV, int NT, bool BF>
__device__ __forceinline__ void qvb8_mma_group_n(
    float (&acc)[NT][4], const uint32_t (&va)[NV], const uint32_t (&vb)[NV],
    int nba, int nbb, uint32_t maska, uint32_t maskb,
    float ma, float mb, float tsa, float tsb, const uint4 (&xq)[NT]) {
    uint32_t loa = 0, hia = 0, lob = 0, hib = 0;
    if constexpr (NB != 0 && NB <= 8) {
        constexpr int WI = (J * NB) >> 2, B = ((J * NB) & 3) * 8;
        loa = __funnelshift_r(va[WI], va[WI + 1], B);  hia = __funnelshift_r(va[WI + 1], va[WI + 2], B);
        lob = __funnelshift_r(vb[WI], vb[WI + 1], B);  hib = __funnelshift_r(vb[WI + 1], vb[WI + 2], B);
    }
    float fa[8], fb[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint32_t ia, ib;
        if constexpr (NB > 8) {
            switch (k) {
                case 0: ia = field_w<NB, J, 0>(va); ib = field_w<NB, J, 0>(vb); break;
                case 1: ia = field_w<NB, J, 1>(va); ib = field_w<NB, J, 1>(vb); break;
                case 2: ia = field_w<NB, J, 2>(va); ib = field_w<NB, J, 2>(vb); break;
                case 3: ia = field_w<NB, J, 3>(va); ib = field_w<NB, J, 3>(vb); break;
                case 4: ia = field_w<NB, J, 4>(va); ib = field_w<NB, J, 4>(vb); break;
                case 5: ia = field_w<NB, J, 5>(va); ib = field_w<NB, J, 5>(vb); break;
                case 6: ia = field_w<NB, J, 6>(va); ib = field_w<NB, J, 6>(vb); break;
                default: ia = field_w<NB, J, 7>(va); ib = field_w<NB, J, 7>(vb); break;
            }
        } else if constexpr (NB != 0) {
            switch (k) {
                case 0: ia = field_of<NB, 0>(loa, hia); ib = field_of<NB, 0>(lob, hib); break;
                case 1: ia = field_of<NB, 1>(loa, hia); ib = field_of<NB, 1>(lob, hib); break;
                case 2: ia = field_of<NB, 2>(loa, hia); ib = field_of<NB, 2>(lob, hib); break;
                case 3: ia = field_of<NB, 3>(loa, hia); ib = field_of<NB, 3>(lob, hib); break;
                case 4: ia = field_of<NB, 4>(loa, hia); ib = field_of<NB, 4>(lob, hib); break;
                case 5: ia = field_of<NB, 5>(loa, hia); ib = field_of<NB, 5>(lob, hib); break;
                case 6: ia = field_of<NB, 6>(loa, hia); ib = field_of<NB, 6>(lob, hib); break;
                default: ia = field_of<NB, 7>(loa, hia); ib = field_of<NB, 7>(lob, hib); break;
            }
        } else {
            const int bita = (J * 8 + k) * nba, bitb = (J * 8 + k) * nbb;
            ia = __funnelshift_r(va[bita >> 5], va[(bita >> 5) + 1], bita & 31) & maska;
            ib = __funnelshift_r(vb[bitb >> 5], vb[(bitb >> 5) + 1], bitb & 31) & maskb;
        }
        fa[k] = fmaf(small_uint_to_float(ia), tsa, ma);
        fb[k] = fmaf(small_uint_to_float(ib), tsb, mb);
    }
    auto h2 = [](float lo, float hi) {
        if constexpr (BF) {
            __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
            return *reinterpret_cast<uint32_t*>(&v);
        } else {
            __half2 v = __floats2half2_rn(lo, hi);
            return *reinterpret_cast<uint32_t*>(&v);
        }
    };
    const uint32_t a0 = h2(fa[0], fa[1]), a1 = h2(fb[0], fb[1]), a2 = h2(fa[2], fa[3]), a3 = h2(fb[2], fb[3]);
    const uint32_t a4 = h2(fa[4], fa[5]), a5 = h2(fb[4], fb[5]), a6 = h2(fa[6], fa[7]), a7 = h2(fb[6], fb[7]);
    #pragma unroll
    for (int t = 0; t < NT; ++t) {
        mma_16816<BF>(acc[t], a0, a1, a2, a3, xq[t].x, xq[t].y);
        mma_16816<BF>(acc[t], a4, a5, a6, a7, xq[t].z, xq[t].w);
    }
}

// Activations of group J (columns 32c + 8J .. +7 of super-tile st) for every token group.
template <int NT>
__device__ __forceinline__ void qvb8_mma_load_x(uint4 (&xq)[NT], const half* __restrict__ X, int P, int M,
                                                int g, int c, int st, int J) {
    #pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int tok = g + 8 * t;
        xq[t] = tok < M ? __ldg(reinterpret_cast<const uint4*>(X + (size_t)tok * P + c * 32 + (size_t)st * 128) + J)
                        : make_uint4(0u, 0u, 0u, 0u);
    }
}

template <int NT>
__device__ __forceinline__ void qvb8_mma_store(float* __restrict__ Y, const float (&acc)[NT][4], int Q, int M,
                                               int c, bool va, bool vb, int ra, int rb) {
    // C fragment of group t: acc[t][0..1] = row g, tokens 8t+2c, 8t+2c+1; acc[t][2..3] = row g+8, same tokens.
    #pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int n0 = 8 * t + 2 * c, n1 = n0 + 1;
        if (va) {
            if (n0 < M) Y[(int64_t)n0 * Q + ra] = acc[t][0];
            if (n1 < M) Y[(int64_t)n1 * Q + ra] = acc[t][1];
        }
        if (vb) {
            if (n0 < M) Y[(int64_t)n0 * Q + rb] = acc[t][2];
            if (n1 < M) Y[(int64_t)n1 * Q + rb] = acc[t][3];
        }
    }
}

// The four 32-column groups of one super-tile. Up to 16 tokens (NT <= 2) every group's activations are loaded
// up front so the loads overlap (the old schedule; loading them group by group cost 441 -> 486 ms at 16 rows);
// with more token groups they are loaded per group to bound registers (NT=6: 24 instead of 96).
#define QVB8_MMA_GROUPS(NBA, NBB, MA_, MB_) do {                                                          \
    if constexpr (NT <= 2) {                                                                              \
        uint4 x0[NT], x1[NT], x2[NT], x3[NT];                                                             \
        qvb8_mma_load_x<NT>(x0, X, P, M, g, c, st, 0); qvb8_mma_load_x<NT>(x1, X, P, M, g, c, st, 1);    \
        qvb8_mma_load_x<NT>(x2, X, P, M, g, c, st, 2); qvb8_mma_load_x<NT>(x3, X, P, M, g, c, st, 3);    \
        qvb8_mma_group_n<NB, 0, NW - 1, NT, BF>(acc, vA, vB, NBA, NBB, MA_, MB_, ma, mb, tsa, tsb, x0);     \
        qvb8_mma_group_n<NB, 1, NW - 1, NT, BF>(acc, vA, vB, NBA, NBB, MA_, MB_, ma, mb, tsa, tsb, x1);     \
        qvb8_mma_group_n<NB, 2, NW - 1, NT, BF>(acc, vA, vB, NBA, NBB, MA_, MB_, ma, mb, tsa, tsb, x2);     \
        qvb8_mma_group_n<NB, 3, NW - 1, NT, BF>(acc, vA, vB, NBA, NBB, MA_, MB_, ma, mb, tsa, tsb, x3);     \
    } else {                                                                                              \
        uint4 xq[NT];                                                                                     \
        qvb8_mma_load_x<NT>(xq, X, P, M, g, c, st, 0);                                                    \
        qvb8_mma_group_n<NB, 0, NW - 1, NT, BF>(acc, vA, vB, NBA, NBB, MA_, MB_, ma, mb, tsa, tsb, xq);     \
        qvb8_mma_load_x<NT>(xq, X, P, M, g, c, st, 1);                                                    \
        qvb8_mma_group_n<NB, 1, NW - 1, NT, BF>(acc, vA, vB, NBA, NBB, MA_, MB_, ma, mb, tsa, tsb, xq);     \
        qvb8_mma_load_x<NT>(xq, X, P, M, g, c, st, 2);                                                    \
        qvb8_mma_group_n<NB, 2, NW - 1, NT, BF>(acc, vA, vB, NBA, NBB, MA_, MB_, ma, mb, tsa, tsb, xq);     \
        qvb8_mma_load_x<NT>(xq, X, P, M, g, c, st, 3);                                                    \
        qvb8_mma_group_n<NB, 3, NW - 1, NT, BF>(acc, vA, vB, NBA, NBB, MA_, MB_, ma, mb, tsa, tsb, xq);     \
    }                                                                                                     \
} while (0)

template <int NB, int NT, bool BF>
__device__ __forceinline__ void qvb8_mma_tile_n(
    const half* __restrict__ X, const uint8_t* __restrict__ ids,
    float* __restrict__ Y, int P, int Q, int M, int g, int c, bool va, bool vb,
    int ra, int rb, int nba_rt, int nbb_rt, int64_t baseA, int64_t baseB,
    float ma, float mb, float tsa, float tsb, int s_begin, int s_end, int64_t ids_bytes) {
    const int nba = NB ? NB : nba_rt, nbb = NB ? NB : nbb_rt;
    const uint32_t maska = (1u << nba) - 1u, maskb = (1u << nbb) - 1u;
    constexpr int NW = QvbMmaWords<NB>::value;
    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    const int64_t last_word = ids_bytes - 4;
    for (int st0 = s_begin; st0 < s_end; st0 += QVB8_MMA_U) {
        uint32_t wa[QVB8_MMA_U][NW], wb[QVB8_MMA_U][NW];
        #pragma unroll
        for (int u = 0; u < QVB8_MMA_U; ++u) {
            const int st = min(st0 + u, s_end - 1);
            const int64_t a0 = (baseA + (int64_t)(st * 16 + c * 4) * nba) & ~(int64_t)3;
            const int64_t b0 = (baseB + (int64_t)(st * 16 + c * 4) * nbb) & ~(int64_t)3;
            #pragma unroll
            for (int i = 0; i < NW; ++i) {
                wa[u][i] = __ldg(reinterpret_cast<const uint32_t*>(ids + min(a0 + 4 * i, last_word)));
                wb[u][i] = __ldg(reinterpret_cast<const uint32_t*>(ids + min(b0 + 4 * i, last_word)));
            }
        }
        #pragma unroll
        for (int u = 0; u < QVB8_MMA_U; ++u) {
            const int st = st0 + u;
            if (st >= s_end) break;                                // warp-uniform
            const uint32_t sha = (uint32_t)((baseA + (int64_t)(st * 16 + c * 4) * nba) & 3) * 8u;
            const uint32_t shb = (uint32_t)((baseB + (int64_t)(st * 16 + c * 4) * nbb) & 3) * 8u;
            uint32_t vA[NW - 1], vB[NW - 1];
            #pragma unroll
            for (int i = 0; i < NW - 1; ++i) {
                vA[i] = __funnelshift_r(wa[u][i], wa[u][i + 1], sha);
                vB[i] = __funnelshift_r(wb[u][i], wb[u][i + 1], shb);
            }
            QVB8_MMA_GROUPS(nba, nbb, maska, maskb);
        }
    }
    qvb8_mma_store<NT>(Y, acc, Q, M, c, va, vb, ra, rb);
}

// v1: shared-memory staged variant of qvb8_mma_tile for fixed-width tiles (NB 4..8).
// The register version has each lane load its own row's words with 4-byte __ldg: one load instruction then
// addresses 8 rows x 4 lanes at a 4*NB-byte stride. Here the warp copies each of its 16 rows' super-tile
// spans into shared memory with consecutive lanes on consecutive words, issuing the NEXT super-tile's loads
// before computing the current one (double buffer), then every lane reads its window from shared memory.
// Decode, mma and output are the register version's (bit-identical). 65B, 16 rows: 471 -> 441 ms / 80 layers.
template <int NB, int NT, bool BF>
__device__ __forceinline__ void qvb8_mma_tile_smem_n(
    const half* __restrict__ X, const uint8_t* __restrict__ ids,
    float* __restrict__ Y, int P, int Q, int M, int g, int c, int lane, bool va, bool vb,
    int ra, int rb, int64_t baseA, int64_t baseB,
    float ma, float mb, float tsa, float tsb, int s_begin, int s_end, int64_t ids_bytes,
    uint32_t* __restrict__ buf, int64_t* __restrict__ sb) {
    constexpr int NW = QvbMmaWords<NB>::value;
    constexpr int RWN = ((3 + 12 * NB) >> 2) + NW + 1;   // words used per row (lane c=3, worst alignment)
    static_assert(RWN <= QVB8_SM_RW, "staged row too wide");
    constexpr int TOTAL = 16 * RWN;
    const uint32_t mask = (1u << NB) - 1u;
    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    const int64_t last_word = ids_bytes - 4;
    if (c == 0) { sb[g] = baseA; sb[g + 8] = baseB; }
    __syncwarp();
    constexpr int NL = (TOTAL + WARP - 1) / WARP;
    uint32_t ld[NL];
    auto issue = [&](int st) {                       // loads of super-tile st into registers
        const int64_t sto = (int64_t)st * 16 * NB;
        #pragma unroll
        for (int k = 0; k < NL; ++k) {
            const int idx = lane + k * WARP;
            const int r = min(idx / RWN, 15), w = idx - (idx / RWN) * RWN;
            const int64_t a = ((sb[r] + sto) & ~(int64_t)3) + 4 * w;
            ld[k] = idx < TOTAL ? __ldg(reinterpret_cast<const uint32_t*>(ids + min(a, last_word))) : 0u;
        }
    };
    auto commit = [&](uint32_t* b) {
        #pragma unroll
        for (int k = 0; k < NL; ++k) {
            const int idx = lane + k * WARP;
            if (idx < TOTAL) b[(idx / RWN) * QVB8_SM_RW + idx - (idx / RWN) * RWN] = ld[k];
        }
    };
    if (s_begin < s_end) { issue(s_begin); commit(buf); }
    __syncwarp();
    for (int st = s_begin; st < s_end; ++st) {
        uint32_t* cur = buf + ((st - s_begin) & 1) * 16 * QVB8_SM_RW;
        uint32_t* nxt = buf + (((st - s_begin) & 1) ^ 1) * 16 * QVB8_SM_RW;
        if (st + 1 < s_end) issue(st + 1);            // next tile's loads fly while this one computes
        const int64_t sto = (int64_t)st * 16 * NB;
        const int offA = (int)((baseA + sto) & 3) + c * 4 * NB, offB = (int)((baseB + sto) & 3) + c * 4 * NB;
        const uint32_t* pa = cur + g * QVB8_SM_RW + (offA >> 2);
        const uint32_t* pb = cur + (g + 8) * QVB8_SM_RW + (offB >> 2);
        const uint32_t sha = (uint32_t)(offA & 3) * 8u, shb = (uint32_t)(offB & 3) * 8u;
        uint32_t vA[NW - 1], vB[NW - 1];
        #pragma unroll
        for (int i = 0; i < NW - 1; ++i) {
            vA[i] = __funnelshift_r(pa[i], pa[i + 1], sha);
            vB[i] = __funnelshift_r(pb[i], pb[i + 1], shb);
        }
        QVB8_MMA_GROUPS(NB, NB, mask, mask);
        if (st + 1 < s_end) commit(nxt);
        __syncwarp();
    }
    qvb8_mma_store<NT>(Y, acc, Q, M, c, va, vb, ra, rb);
}

// v1: fixed-width tiles only (NB 4..8, the shared-memory path). The combined kernel's register count is
// set by its rare mixed-width fallback (255 registers/thread -> 2 blocks = 8 warps per SM, too few loads in
// flight for the memory latency: ~127 GB/s). Without that path, and with a minimum-blocks bound, more warps
// stay resident. Mixed-width tiles (few) still run in qvb8_mma_multi with mode=1.
#ifndef QVB8_MMA_MINB
#define QVB8_MMA_MINB 3
#endif
template <int WARPS, int NT, bool BF>
__global__ void __launch_bounds__(WARPS * WARP, QVB8_MMA_MINB) qvb8_mma_multi_fixed(
    const half* __restrict__ X, const uint8_t* __restrict__ ids, const float* __restrict__ min_s,
    const uint8_t* __restrict__ row_nbits, const int64_t* __restrict__ row_bit_off, float* __restrict__ Y,
    const float* __restrict__ tau_s, const int* __restrict__ perm, const int* __restrict__ tile_nb,
    int P, int Q, int M, int64_t ids_bytes, int mode) {
    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int g = lane >> 2, c = lane & 3;
    const int tile = blockIdx.x * WARPS + warp;
    const int q0 = tile * 16;
    if (q0 >= Q) return;
    const int nbt = tile_nb[tile];
    if (nbt < 4 || nbt > 12) return;                  // mixed (0) or another width: the general kernel
    const int pa = q0 + g, pb = q0 + g + 8;
    const bool va = pa < Q, vb = pb < Q;
    const int last = perm[Q - 1];
    const int ra = va ? perm[pa] : last;
    const int rb = vb ? perm[pb] : last;
    const int64_t baseA = row_bit_off[ra] >> 3, baseB = row_bit_off[rb] >> 3;
    const float ma = min_s[ra], mb = min_s[rb], tsa = tau_s[ra], tsb = tau_s[rb];
    const int n_super = P / 128;
    const int per = (n_super + (int)gridDim.y - 1) / (int)gridDim.y;
    const int s_begin = (int)blockIdx.y * per;
    const int s_end = min(n_super, s_begin + per);
    Y += (size_t)blockIdx.y * M * Q;
    __shared__ uint32_t smem_buf[WARPS][2 * 16 * (QVB8_SM_RWT > QVB8_SM_RW ? QVB8_SM_RWT : QVB8_SM_RW)];
    __shared__ int64_t smem_base[WARPS][16];
#define QVB8_MMA_CALL_F(NBV) do { if constexpr (NT <= 2) qvb8_mma_tile_smem<NBV, NT, BF>(X, ids, Y, P, Q, M, g, c, \
        lane, va, vb, ra, rb, baseA, baseB, ma, mb, tsa, tsb, s_begin, s_end, ids_bytes, smem_buf[warp], \
        smem_base[warp]); \
    else qvb8_mma_tile_smem_n<NBV, NT, BF>(X, ids, Y, P, Q, M, g, c, lane, va, vb, ra, rb, \
        baseA, baseB, ma, mb, tsa, tsb, s_begin, s_end, ids_bytes, smem_buf[warp], smem_base[warp]); } while (0)
    switch (nbt) {
        case 4: QVB8_MMA_CALL_F(4); break;
        case 5: QVB8_MMA_CALL_F(5); break;
        case 6: QVB8_MMA_CALL_F(6); break;
        case 7: QVB8_MMA_CALL_F(7); break;
        case 8: QVB8_MMA_CALL_F(8); break;
        case 9: QVB8_MMA_CALL_F(9); break;
        case 10: QVB8_MMA_CALL_F(10); break;
        case 11: QVB8_MMA_CALL_F(11); break;
        default: QVB8_MMA_CALL_F(12); break;
    }
#undef QVB8_MMA_CALL_F
}

// Whether a weight has a tile the lean kernel does not take (mixed = 0, or 1-3 / 9-14 bits). Host sync: callers
// pass it precomputed (general >= 0); this is the fallback.
static bool qvb8_has_general_tiles(const torch::Tensor& tile_nb) {
    return ((tile_nb < 4) | (tile_nb > 12)).any().item<bool>();
}

template <int WARPS, int NT, bool BF>
__global__ void __launch_bounds__(WARPS * WARP) qvb8_mma_multi(
    const half*    __restrict__ X,          // (M, P) fp16
    const uint8_t* __restrict__ ids,
    const float*   __restrict__ min_s,
    const uint8_t* __restrict__ row_nbits,
    const int64_t* __restrict__ row_bit_off,
    float*         __restrict__ Y,          // (S, M, Q)
    const float*   __restrict__ tau_s,
    const int*     __restrict__ perm,       // sorted position -> row, or null
    const int*     __restrict__ tile_nb,    // per 16-row tile: its width, 0 = mixed; or null
    int P, int Q, int M, int64_t ids_bytes, int mode = 0) {

    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int g = lane >> 2, c = lane & 3;
    const int tile = blockIdx.x * WARPS + warp;
    const int q0 = tile * 16;
    if (q0 >= Q) return;
    if (mode == 1 && tile_nb && tile_nb[tile] >= 4 && tile_nb[tile] <= 12) return;  // done by qvb8_mma_multi_fixed
    const int pa = q0 + g, pb = q0 + g + 8;
    const bool va = pa < Q, vb = pb < Q;
    const int last = perm ? perm[Q - 1] : Q - 1;                 // invalid rows read a real row; their results are dropped
    const int ra = va ? (perm ? perm[pa] : pa) : last;
    const int rb = vb ? (perm ? perm[pb] : pb) : last;
    const int nba = row_nbits[ra], nbb = row_nbits[rb];
    const int64_t baseA = row_bit_off[ra] >> 3, baseB = row_bit_off[rb] >> 3;
    const float ma = min_s[ra], mb = min_s[rb], tsa = tau_s[ra], tsb = tau_s[rb];

    const int n_super = P / 128;
    const int per = (n_super + (int)gridDim.y - 1) / (int)gridDim.y;
    const int s_begin = (int)blockIdx.y * per;
    const int s_end = min(n_super, s_begin + per);
    Y += (size_t)blockIdx.y * M * Q;

#define QVB8_MMA_CALL(NBV) do { if constexpr (NT <= 2) qvb8_mma_tile<NBV, NT, BF>(X, ids, Y, P, Q, M, g, c, va, vb, \
        ra, rb, nba, nbb, baseA, baseB, ma, mb, tsa, tsb, s_begin, s_end, ids_bytes); \
    else qvb8_mma_tile_n<NBV, NT, BF>(X, ids, Y, P, Q, M, g, c, va, vb, ra, rb, nba, nbb, \
        baseA, baseB, ma, mb, tsa, tsb, s_begin, s_end, ids_bytes); } while (0)
#if QVB8_MMA_SMEM
    __shared__ uint32_t smem_buf[WARPS][2 * 16 * (QVB8_SM_RWT > QVB8_SM_RW ? QVB8_SM_RWT : QVB8_SM_RW)];
    __shared__ int64_t smem_base[WARPS][16];
#define QVB8_MMA_CALL_S(NBV) do { if constexpr (NT <= 2) qvb8_mma_tile_smem<NBV, NT, BF>(X, ids, Y, P, Q, M, g, c, \
        lane, va, vb, ra, rb, baseA, baseB, ma, mb, tsa, tsb, s_begin, s_end, ids_bytes, smem_buf[warp], \
        smem_base[warp]); \
    else qvb8_mma_tile_smem_n<NBV, NT, BF>(X, ids, Y, P, Q, M, g, c, lane, va, vb, ra, rb, \
        baseA, baseB, ma, mb, tsa, tsb, s_begin, s_end, ids_bytes, smem_buf[warp], smem_base[warp]); } while (0)
    switch (tile_nb ? tile_nb[tile] : 0) {          // warp-uniform
        case 4: QVB8_MMA_CALL_S(4); break;
        case 5: QVB8_MMA_CALL_S(5); break;
        case 6: QVB8_MMA_CALL_S(6); break;
        case 7: QVB8_MMA_CALL_S(7); break;
        case 8: QVB8_MMA_CALL_S(8); break;
        default: QVB8_MMA_CALL(0); break;
    }
#undef QVB8_MMA_CALL_S
#else
    switch (tile_nb ? tile_nb[tile] : 0) {          // warp-uniform
        case 4: QVB8_MMA_CALL(4); break;
        case 5: QVB8_MMA_CALL(5); break;
        case 6: QVB8_MMA_CALL(6); break;
        case 7: QVB8_MMA_CALL(7); break;
        case 8: QVB8_MMA_CALL(8); break;
        default: QVB8_MMA_CALL(0); break;
    }
#endif
#undef QVB8_MMA_CALL
}

// Escapes are weights stored exactly because they fell outside their row's band; the main pass above
// multiplied the BANDED value in their place. Correct afterwards, as the GEMV path does: add
// X[t][c] * (restore(exact) - restore(banded)) per token, where restore() is the same single rounding
// the main pass applied (fp16 or bf16), so the result equals what the decoded matrix would give.
template <bool BF>
__global__ void qvb8_mma_fix_escapes(
    const half*    __restrict__ X,            // (M, P), fp16 or bf16 bits
    const uint8_t* __restrict__ ids,
    const float*   __restrict__ min_s,
    const uint8_t* __restrict__ row_nbits,
    const int64_t* __restrict__ row_bit_off,
    const int64_t* __restrict__ esc_idx,      // flat position q*P + c
    const float*   __restrict__ esc_vals,
    float*         __restrict__ Y,            // (M, Q)
    const float*   __restrict__ tau_s,
    int P, int Q, int M, int64_t n_esc) {
    // esc_idx is sorted (QVBandLinearV8): the thread at the first escape of an output q adds all of q's escapes in
    // order for each token, then writes once -- no atomics, bit-reproducible
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_esc) return;
    const int64_t q = esc_idx[i] / P;
    if (i > 0 && esc_idx[i - 1] / P == q) return;
    int64_t e1 = i + 1;
    while (e1 < n_esc && esc_idx[e1] / P == q) ++e1;
    const int nb = row_nbits[q];
    // Each escape's correction (restore(exact) - restore(banded)) is decoded ONCE and applied to every token
    // (M <= 16, accumulators in registers). Per token the escapes are still summed in ascending order, so the
    // result is bit-identical to the old token-outer loop, which re-decoded every escape M times.
    float a[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) a[t] = 0.f;
    for (int64_t e = i; e < e1; ++e) {
        const int c = (int)(esc_idx[e] - q * P);
        const uint32_t id = load_field(ids, row_bit_off[q] + (int64_t)c * nb, nb);
        const float banded = fmaf((float)id, tau_s[q], min_s[q]);
        float wb, we;
        if constexpr (BF) { wb = __bfloat162float(__float2bfloat16_rn(banded)); we = __bfloat162float(__float2bfloat16_rn(esc_vals[e])); }
        else              { wb = __half2float(__float2half_rn(banded));         we = __half2float(__float2half_rn(esc_vals[e])); }
        const float d = we - wb;
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (t < M) {
                float x;
                if constexpr (BF) x = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(X)[(size_t)t * P + c]);
                else              x = __half2float(X[(size_t)t * P + c]);
                a[t] += x * d;
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < 16; ++t)
        if (t < M) Y[(int64_t)t * Q + q] += a[t];
}

// Row-parallel escape fix (v1): one WARP per output row that has escapes. The old kernel gave each
// such row to ONE thread that walked all of its escapes serially; a row with thousands of escapes (the 3e-3
// DFlash2 drafter) then ran one thread for ~0.6 ms per linear -- 38% of a draft step. Here the lanes split the
// row's escapes (lane l takes escapes l, l+32, ...), each keeps 16 per-token partial sums, and a fixed xor-
// shuffle tree combines them: deterministic (same order every run), not bit-identical to the serial sum.
// row_start[r]..row_start[r+1] are the escapes of the r-th row that has any (esc_idx is sorted).
template <bool BF>
__global__ void qvb8_mma_fix_escapes_rows(
    const half*    __restrict__ X, const uint8_t* __restrict__ ids, const float* __restrict__ min_s,
    const uint8_t* __restrict__ row_nbits, const int64_t* __restrict__ row_bit_off,
    const int64_t* __restrict__ esc_idx, const float* __restrict__ esc_vals,
    const int64_t* __restrict__ row_start, float* __restrict__ Y, const float* __restrict__ tau_s,
    int P, int Q, int M, int n_rows) {
    const int r = (int)(((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / WARP);
    const int lane = threadIdx.x % WARP;
    if (r >= n_rows) return;                                   // warp-uniform
    const int64_t e0 = row_start[r], e1 = row_start[r + 1];
    const int64_t q = esc_idx[e0] / P;
    const int nb = row_nbits[q];
    const int64_t boff = row_bit_off[q];
    const float tq = tau_s[q], mq = min_s[q];
    float a[48];
    #pragma unroll
    for (int t = 0; t < 48; ++t) a[t] = 0.f;
    for (int64_t e = e0 + lane; e < e1; e += WARP) {
        const int c = (int)(esc_idx[e] - q * P);
        const uint32_t id = load_field(ids, boff + (int64_t)c * nb, nb);
        const float banded = fmaf((float)id, tq, mq);
        float wb, we;
        if constexpr (BF) { wb = __bfloat162float(__float2bfloat16_rn(banded)); we = __bfloat162float(__float2bfloat16_rn(esc_vals[e])); }
        else              { wb = __half2float(__float2half_rn(banded));         we = __half2float(__float2half_rn(esc_vals[e])); }
        const float d = we - wb;
        #pragma unroll
        for (int t = 0; t < 48; ++t) {
            if (t < M) {
                float x;
                if constexpr (BF) x = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(X)[(size_t)t * P + c]);
                else              x = __half2float(X[(size_t)t * P + c]);
                a[t] = fmaf(x, d, a[t]);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < 48; ++t) {
        if (t < M) {
            float v = a[t];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) Y[(int64_t)t * Q + q] += v;
        }
    }
}

// Start offsets of each escaped row's run in a sorted esc_idx (+ a final n). Needs a host sync, so callers
// compute it ONCE per weight and pass it in (QVBandLinearV8._mma_order); this is only the fallback for a caller
// that does not. (An earlier version cached by device address: a freed weight's slot reused by another weight
// of the same size would have handed the kernel stale offsets, and the cache kept every tensor alive.)
static torch::Tensor qvb8_esc_row_starts(const torch::Tensor& esc_idx, int64_t P) {
    auto rows = esc_idx.div(P, "floor");
    const int64_t n = rows.numel();
    auto brk = torch::ones({n}, rows.options().dtype(torch::kBool));
    if (n > 1) brk.slice(0, 1).copy_(rows.slice(0, 1) != rows.slice(0, 0, n - 1));
    return torch::cat({torch::nonzero(brk).view(-1), torch::full({1}, n, rows.options())}).contiguous();
}

static bool qvb8_mma_ok(const torch::Tensor& ids, int64_t P) {
    return (P % 128) == 0 && (reinterpret_cast<uintptr_t>(ids.data_ptr<uint8_t>()) & 3u) == 0;
}

torch::Tensor qvb8_mma_multi_forward(torch::Tensor X, torch::Tensor ids, torch::Tensor min_s,
                                     torch::Tensor row_nbits, torch::Tensor row_bit_off, int64_t P,
                                     torch::Tensor tau_s, torch::Tensor esc_idx,
                                     torch::Tensor esc_vals, torch::Tensor perm,
                                     torch::Tensor tile_nb, bool bf16, c10::optional<torch::Tensor> esc_starts,
                                     int64_t general) {
    // Launch on the card the operands live on, not whichever is current:
    // a wrong current device reads another GPU's memory (Xid 31).
    const c10::cuda::OptionalCUDAGuard _qv_guard(device_of(X));
    // mma.m16n8k16 is sm_80+; below that the kernel body is __trap(), which would poison the whole CUDA
    // context -- refuse with a normal error instead (callers pick the GEMV path on pre-Ampere GPUs).
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8,
                "qvb8 mma-multi: needs sm_80+ (this GPU is sm_", at::cuda::getCurrentDeviceProperties()->major,
                at::cuda::getCurrentDeviceProperties()->minor, "); use forward / gemv");
    const int64_t Q = min_s.numel(), M = X.size(0);
    TORCH_CHECK(M >= 1 && M <= 48, "qvb8 mma-multi: M=", M, " outside 1..48");
    TORCH_CHECK(qvb8_mma_ok(ids, P), "qvb8 mma-multi: needs P%128==0 and 4-byte-aligned ids");
    const int64_t n_esc = esc_idx.numel();
    TORCH_CHECK(n_esc == 0 || (esc_idx.scalar_type() == torch::kLong && esc_vals.scalar_type() == torch::kFloat
                               && esc_vals.numel() == n_esc),
                "qvb8 mma-multi: escapes must be int64 positions and float32 values");
    TORCH_CHECK(perm.numel() == 0 || (perm.numel() == Q && perm.scalar_type() == torch::kInt && perm.is_cuda()),
                "qvb8 mma-multi: perm must be an int32 CUDA tensor of length Q (or empty)");
    TORCH_CHECK(tile_nb.numel() == 0 || (perm.numel() == Q && tile_nb.numel() == (Q + 15) / 16
                                         && tile_nb.scalar_type() == torch::kInt && tile_nb.is_cuda()),
                "qvb8 mma-multi: tile_nb needs perm and one int32 entry per 16 rows");
    // fp16 operands for an fp16 pack (activations saturated to fp16's range); bf16 operands for a
    // bf16-certified pack (bf16 has fp32's range, so the plain conversion cannot overflow).
    auto Xh = bf16 ? X.to(torch::kBFloat16).contiguous() : saturating_half(X);
    auto Y = torch::empty({M, Q}, torch::dtype(torch::kFloat).device(X.device()));
    constexpr int WARPS = 4;
    const int blocks = (int)((Q + 16 * WARPS - 1) / (16 * WARPS));
    // Few output tiles (down_proj: 5120 rows = 320 warps) cannot fill the part: split K.
    int S = 1;
    const int64_t tiles = (Q + 15) / 16;
    // 2048 (was 4096): measured on the dev GPU, 65B layer, 16 rows 520 -> 482 ms per 80 layers; fewer partials to combine
    while (tiles * S < 2048 && S < 8 && (P / 128) / (S * 2) >= 2 * QVB8_MMA_U) S *= 2;
    torch::Tensor part;
    float* ydst = Y.data_ptr<float>();
    if (S > 1) {
        part = torch::empty({(int64_t)S * M, Q}, torch::dtype(torch::kFloat).device(X.device()));
        ydst = part.data_ptr<float>();
    }
    // v1: fixed-width tiles in the lean kernel, mixed-width tiles (if any) in the general one.
    const bool split = QVB8_MMA_SMEM && tile_nb.numel() > 0;
    const bool mixed = split && (general >= 0 ? general != 0 : qvb8_has_general_tiles(tile_nb));
    auto launch = [&](auto kern, int mode) {
        kern<<<dim3(blocks, S), WARPS * WARP, 0, at::cuda::getCurrentCUDAStream()>>>(
            (const half*)Xh.data_ptr(), ids.data_ptr<uint8_t>(), min_s.data_ptr<float>(),
            row_nbits.data_ptr<uint8_t>(), row_bit_off.data_ptr<int64_t>(), ydst,
            tau_s.data_ptr<float>(), perm.numel() ? perm.data_ptr<int>() : nullptr,
            tile_nb.numel() ? tile_nb.data_ptr<int>() : nullptr, (int)P, (int)Q, (int)M, ids.numel(), mode);
    };
    auto run = [&](auto kern) { launch(kern, split ? 1 : 0); };
    auto run_fixed = [&](auto kern) { if (split) launch(kern, 0); };
    // token groups of 8: decoded weights are shared by all of them (v1: up to 48 tokens)
    const int nt = (int)((M + 7) / 8);
    if (split) {
        if (bf16) {
            switch (nt) {
                case 1: run_fixed(qvb8_mma_multi_fixed<WARPS, 1, true>); break;  case 2: run_fixed(qvb8_mma_multi_fixed<WARPS, 2, true>); break;
                case 3: run_fixed(qvb8_mma_multi_fixed<WARPS, 3, true>); break;  case 4: run_fixed(qvb8_mma_multi_fixed<WARPS, 4, true>); break;
                case 5: run_fixed(qvb8_mma_multi_fixed<WARPS, 5, true>); break;  default: run_fixed(qvb8_mma_multi_fixed<WARPS, 6, true>); break;
            }
        } else {
            switch (nt) {
                case 1: run_fixed(qvb8_mma_multi_fixed<WARPS, 1, false>); break; case 2: run_fixed(qvb8_mma_multi_fixed<WARPS, 2, false>); break;
                case 3: run_fixed(qvb8_mma_multi_fixed<WARPS, 3, false>); break; case 4: run_fixed(qvb8_mma_multi_fixed<WARPS, 4, false>); break;
                case 5: run_fixed(qvb8_mma_multi_fixed<WARPS, 5, false>); break; default: run_fixed(qvb8_mma_multi_fixed<WARPS, 6, false>); break;
            }
        }
    }
    if (!split || mixed) {
    if (bf16) {
        switch (nt) {
            case 1: run(qvb8_mma_multi<WARPS, 1, true>); break;  case 2: run(qvb8_mma_multi<WARPS, 2, true>); break;
            case 3: run(qvb8_mma_multi<WARPS, 3, true>); break;  case 4: run(qvb8_mma_multi<WARPS, 4, true>); break;
            case 5: run(qvb8_mma_multi<WARPS, 5, true>); break;  default: run(qvb8_mma_multi<WARPS, 6, true>); break;
        }
    } else {
        switch (nt) {
            case 1: run(qvb8_mma_multi<WARPS, 1, false>); break; case 2: run(qvb8_mma_multi<WARPS, 2, false>); break;
            case 3: run(qvb8_mma_multi<WARPS, 3, false>); break; case 4: run(qvb8_mma_multi<WARPS, 4, false>); break;
            case 5: run(qvb8_mma_multi<WARPS, 5, false>); break; default: run(qvb8_mma_multi<WARPS, 6, false>); break;
        }
    }
    }
    if (S > 1) {
        const int64_t n = M * Q;
        qvb8_splitk_combine<<<(int)((n + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            ydst, Y.data_ptr<float>(), n, S);
    }
    if (n_esc > 0) {
        auto starts = (esc_starts.has_value() && esc_starts->defined() && esc_starts->numel()) ? *esc_starts
                                                                                       : qvb8_esc_row_starts(esc_idx, P);
        const int n_rows = (int)starts.numel() - 1;
        const int T = 256, WPB = T / WARP;
        auto fix = bf16 ? qvb8_mma_fix_escapes_rows<true> : qvb8_mma_fix_escapes_rows<false>;
        fix<<<(n_rows + WPB - 1) / WPB, T, 0, at::cuda::getCurrentCUDAStream()>>>(
            (const half*)Xh.data_ptr(), ids.data_ptr<uint8_t>(), min_s.data_ptr<float>(),
            row_nbits.data_ptr<uint8_t>(), row_bit_off.data_ptr<int64_t>(), esc_idx.data_ptr<int64_t>(),
            esc_vals.data_ptr<float>(), starts.data_ptr<int64_t>(), Y.data_ptr<float>(), tau_s.data_ptr<float>(),
            (int)P, (int)Q, (int)M, n_rows);
    }
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma_multi", &qvb8_mma_multi_forward, "QVB8 tensor-core multi-row GEMV (M<=48), bf16",
          py::arg("X"), py::arg("ids"), py::arg("min_s"), py::arg("row_nbits"), py::arg("row_bit_off"), py::arg("P"),
          py::arg("tau_s"), py::arg("esc_idx"), py::arg("esc_vals"), py::arg("perm"), py::arg("tile_nb"),
          py::arg("bf16") = false, py::arg("esc_starts") = py::none(), py::arg("general") = -1);
    m.def("forward", &qvb8_forward, "QVB8 forward (any M)");
    m.def("forward_half", &qvb8_forward_half,
          "QVB8 forward; the M>8 GEMM result is stored fp16");
    m.def("gemv", &qvb8_gemv_forward, "QVB8 GEMV (M=1)");
    m.def("decode", &qvb8_decode, "QVB8 full dense decode");
    m.def("decode_bf16", &qvb8_decode_bf16,
          "QVB8 full dense decode to bf16 (uncertified precision; opt-in)");
    m.def("gemv_multi", &qvb8_gemv_multi_forward, "QVB8 multi-row GEMV");
    m.def("gemv_rb", &qvb8_gemv_rb_forward, "QVB8 register-blocked multi-row GEMV (sm_70, M<=8, sorted rows)");
    m.def("moe_experts", &qvb8_moe_experts_forward,
          "Fused MoE expert stack for decode (stacked packs, no escapes)");
    m.def("moe_experts_relu2", &qvb8_moe_experts_relu2_forward,
          "fused non-gated relu^2 MoE expert stack (Nemotron-H)");
}
