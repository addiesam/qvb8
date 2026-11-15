// FlashAttention-style forward for Volta (sm_70): fp16 Q/K/V, fp32 accumulate, fp16 out, head dim 128 or 256, optional causal, GQA.
//
// Why: on a V100 the stock memory-efficient SDPA kernel reaches ~28-33 TFLOPS while a plain fp16 GEMM reaches ~78, and
// FlashAttention / Triton do not use Volta's tensor cores (Triton tl.dot measured 0.2 TFLOPS on sm_70). This kernel runs
// both products (S = Q K^T and O += P V) on the tensor cores through WMMA 16x16x16.
//
// Layout: one CTA = 4 warps = 64 query rows (16 per warp) of one (batch, head); K / V tiles of BN = 64 keys are staged in
// shared memory and shared by the 4 warps. The running output O (16 x 128 fp32 per warp) lives in WMMA accumulator
// fragments for the whole key loop -- no shared-memory round trip per tile.
//
// Softmax: a per-row REFERENCE max m_ref instead of the running max. p = exp(s - m_ref) is exact as long as
// s - m_ref <= 8 (p <= e^8 ~ 2981, stored in fp16: fine); only when a tile's max exceeds m_ref + 8 is O rescaled (spilled
// to shared memory, scaled per row, reloaded) and m_ref moved. With RMS-normalised q / k (Z-Image, Qwen-Image, MiniMax-H3)
// that happens a handful of times per row. The result is the softmax: out = sum_j p_j v_j / sum_j p_j for any m_ref.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

namespace {
constexpr int WARPS = 8;
constexpr int BM = 16 * WARPS;  // query rows per CTA (16 per warp)
constexpr float THRESH = 8.0f;  // move m_ref when a score exceeds it by more than this (log2 domain)

// Per head dim D: BN keys per tile (D=128: 32, D=256: 16 -- the same tensor-core work per tile and the same register
// budget: Q and O fragments stay in registers). Padded strides: KLD (halves) for K / V / Q tiles, PLD (halves) for P,
// SLD (floats) for the S scratch. SLD = BN + 4: rows land 4 banks apart (a stride of 32 floats put all 16 rows in ONE
// bank: 16-way conflicts made the softmax a third of the kernel).
// 1 CTA (8 warps) per SM: ~204 registers / thread at D=128 (2 CTAs forced 128 and spilled).
#define ATTN_CONSTS                                                                 \
  constexpr int KLD = D + 8, PLD = BN + 8, SLD = BN + 4;                            \
  constexpr int PF = BN * (D / 8) / (WARPS * 32); /* uint4 per thread per K (and V) tile */                \
  constexpr int QR = 8;                 /* Q fragments held in registers; the rest are read from shared */   \
  constexpr int QX = D / 16 - QR;       /* D=256: 8 fragments per warp in shared (else 256 spills) */        \
  constexpr int QSL = QX * 16 + 8;      /* row stride (halves) of that per-warp Q copy */

template <int D, int BN>
constexpr size_t smem_bytes() {
  ATTN_CONSTS
  return (size_t)4 * BN * KLD * sizeof(half) + (size_t)WARPS * 16 * SLD * sizeof(float)
       + (size_t)WARPS * 16 * PLD * sizeof(half) + (QX ? (size_t)WARPS * 16 * QSL * sizeof(half) : 0);
}

template <int D, int BN>
__device__ __forceinline__ void scores(wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> (&qa)[8],
                                       const half* Qx, const half* Ks, float* scratch) {
  ATTN_CONSTS
  for (int n = 0; n < BN / 16; ++n) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
    wmma::fill_fragment(s, 0.0f);
#pragma unroll
    for (int k = 0; k < D / 16; ++k) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
      wmma::load_matrix_sync(kb, Ks + n * 16 * KLD + k * 16, KLD);   // B[d][j] = K[j][d]: col-major view of K rows
      if (k < QR) {
        wmma::mma_sync(s, qa[k], kb, s);
      } else {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qx;
        wmma::load_matrix_sync(qx, Qx + (k - QR) * 16, QSL);
        wmma::mma_sync(s, qx, kb, s);
      }
    }
    wmma::store_matrix_sync(scratch + n * 16, s, SLD, wmma::mem_row_major);
  }
}

template <int D, int BN>
__global__ void __launch_bounds__(WARPS * 32, 1)
attn_fwd(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O,
         int Sq, int Sk, float scale_log2, int H, int G, int causal, long qsb, long qsh, long qss, long ksb, long ksh, long kss,
         long vsb, long vsh, long vss, long osb, long osh, long oss) {
  // strides in elements (any layout whose last dim is contiguous: (B, H, S, D) or diffusers' (B, S, H, D) -- no copies)
  ATTN_CONSTS
  static_assert(PF * WARPS * 32 == BN * (D / 8), "K/V tile must split evenly over the threads");
  static_assert(4 * 16 <= 4 * BN, "Q staging: 4 warps x 16 rows must fit the K + V tile buffers");
  static_assert(WARPS % 4 == 0, "Q staging rounds of 4 warps");
  static_assert(D % 32 == 0 && D % BN == 0 && BN % 16 == 0, "tile shapes");
  extern __shared__ __align__(16) unsigned char smem[];
  half* Kbuf = reinterpret_cast<half*>(smem);         // two K tiles, then two V tiles: tile j+1 is written while j computes
  half* Vbuf = Kbuf + 2 * BN * KLD;
  half* Ks = Kbuf;
  float* scratch_all = reinterpret_cast<float*>(Vbuf + 2 * BN * KLD);
  half* P_all = reinterpret_cast<half*>(scratch_all + WARPS * 16 * SLD);

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  float* scratch = scratch_all + warp * 16 * SLD;
  half* Ps = P_all + warp * 16 * PLD;
  half* Qx = P_all + WARPS * 16 * PLD + warp * 16 * QSL;   // this warp's Q columns past the register fragments (D=256)
  const long b = blockIdx.y / H, hh = blockIdx.y % H;
  const half* Qb = Q + b * qsb + hh * qsh;
  const half* Kb = K + b * ksb + (hh / G) * ksh;        // GQA: G query heads share one kv head
  const half* Vb = V + b * vsb + (hh / G) * vsh;
  // causal: heaviest row blocks first (they have the most key tiles), so the tail of the grid is short ones
  const int qblk = causal ? gridDim.x - 1 - blockIdx.x : blockIdx.x;
  const int q0 = qblk * BM + warp * 16;                 // this warp's first query row
  const int off = Sk - Sq;                              // causal is bottom-right aligned: row i sees keys <= i + off
  const int jend = causal ? min(Sk, qblk * BM + BM + off) : Sk;   // keys this CTA can see at all

  // ---- Q (16 x D per warp) -> fragments, staged through the K/V area (zeros past Sq), 4 warps per round
  static_assert(QX >= 0, "QR fragments at most D / 16");
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qa[QR];
  for (int round = 0; round < WARPS / 4; ++round) {
    __syncthreads();
    if (warp / 4 == round) {
      half* qs = Ks + (warp % 4) * 16 * KLD;
      for (int i = lane; i < 16 * (D / 8); i += 32) {
        int r = i / (D / 8), c = (i % (D / 8)) * 8;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (q0 + r < Sq) val = *reinterpret_cast<const uint4*>(Qb + (long)(q0 + r) * qss + c);
        *reinterpret_cast<uint4*>(qs + r * KLD + c) = val;
      }
      __syncwarp();
#pragma unroll
      for (int k = 0; k < QR; ++k) wmma::load_matrix_sync(qa[k], qs + k * 16, KLD);
      for (int i = lane; i < 16 * QX * 2; i += 32) {      // the rest -> this warp's own shared copy (uint4 = 8 halves)
        int rr = i / (QX * 2), c = (i % (QX * 2)) * 8;
        *reinterpret_cast<uint4*>(Qx + rr * QSL + c) = *reinterpret_cast<const uint4*>(qs + rr * KLD + QR * 16 + c);
      }
      __syncwarp();
    }
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[D / 16];
#pragma unroll
  for (int n = 0; n < D / 16; ++n) wmma::fill_fragment(oacc[n], 0.0f);

  // two lanes per row: lane 2r and 2r+1 own row r; they interleave its BN columns in pairs (4i + 2h, 4i + 2h + 1)
  const int r = lane >> 1, h = lane & 1;
  float* srow = scratch + r * SLD;
  float m_ref = -INFINITY, l = 0.0f;
  const int lim = causal ? min(Sk, q0 + r + off + 1) : Sk;   // this row's visible keys: [0, lim)

  // register prefetch (no cp.async on sm_70): tile j+1's K / V are loaded into registers before tile j computes and
  // stored to the other buffer after it, so the global-load latency hides behind the tensor-core work
  uint4 pk[PF], pv[PF];
  auto fetch = [&](int j) {
#pragma unroll
    for (int t = 0; t < PF; ++t) {
      int i = threadIdx.x + t * WARPS * 32, rr = i / (D / 8), c = (i % (D / 8)) * 8;
      pk[t] = pv[t] = make_uint4(0, 0, 0, 0);
      if (j + rr < Sk) {
        pk[t] = *reinterpret_cast<const uint4*>(Kb + (long)(j + rr) * kss + c);
        pv[t] = *reinterpret_cast<const uint4*>(Vb + (long)(j + rr) * vss + c);
      }
    }
  };
  auto put = [&](int buf) {
#pragma unroll
    for (int t = 0; t < PF; ++t) {
      int i = threadIdx.x + t * WARPS * 32, rr = i / (D / 8), c = (i % (D / 8)) * 8;
      *reinterpret_cast<uint4*>(Kbuf + buf * BN * KLD + rr * KLD + c) = pk[t];
      *reinterpret_cast<uint4*>(Vbuf + buf * BN * KLD + rr * KLD + c) = pv[t];
    }
  };
  fetch(0);
  __syncthreads();                                      // the Q staging (in tile buffer 0) is consumed
  put(0);
  __syncthreads();
  int buf = 0;
  for (int j0 = 0; j0 < jend; j0 += BN) {
    if (j0 + BN < jend) fetch(j0 + BN);
    const half* Ks = Kbuf + buf * BN * KLD;
    const half* Vs = Vbuf + buf * BN * KLD;

    scores<D, BN>(qa, Qx, Ks, scratch);
    __syncwarp();

    float sv[BN / 2];
    float mx = -INFINITY;
#pragma unroll
    for (int c = 0; c < BN / 2; ++c) {
      const int col = 2 * (c & ~1) + 2 * h + (c & 1);   // lanes of a row interleave column PAIRS (4i + 2h, 4i + 2h + 1)
      sv[c] = (j0 + col < lim) ? srow[col] * scale_log2 : -INFINITY;
      mx = fmaxf(mx, sv[c]);
    }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    const bool move = mx > m_ref + THRESH;
    if (__any_sync(0xffffffffu, move)) {               // rare: rescale the O rows whose reference moves
      const float alpha = move ? exp2f(m_ref - mx) : 1.0f;   // first tile: m_ref = -inf, O = 0, alpha = 0
      __syncwarp();
#pragma unroll
      for (int n = 0; n < D / 16; n += BN / 16) {      // O through the S scratch, BN columns at a time
#pragma unroll
        for (int t = 0; t < BN / 16; ++t)
          wmma::store_matrix_sync(scratch + t * 16, oacc[n + t], SLD, wmma::mem_row_major);
        __syncwarp();
        for (int c = 0; c < BN / 2; ++c) srow[2 * (c & ~1) + 2 * h + (c & 1)] *= alpha;
        __syncwarp();
#pragma unroll
        for (int t = 0; t < BN / 16; ++t)
          wmma::load_matrix_sync(oacc[n + t], scratch + t * 16, SLD, wmma::mem_row_major);
        __syncwarp();
      }
      if (move) { l *= alpha; m_ref = mx; }
    }
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < BN / 2; c += 2) {
      half2 p = __floats2half2_rn(exp2f(sv[c] - m_ref), exp2f(sv[c + 1] - m_ref));   // masked: exp2(-inf) = 0
      *reinterpret_cast<half2*>(Ps + r * PLD + 2 * c + 2 * h) = p;
      float2 pf = __half22float2(p);
      sum += pf.x + pf.y;                                   // the denominator uses the same rounded p
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    l += sum;
    __syncwarp();

    // ---- O += P V
#pragma unroll
    for (int k = 0; k < BN / 16; ++k) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa;
      wmma::load_matrix_sync(pa, Ps + k * 16, PLD);
#pragma unroll
      for (int n = 0; n < D / 16; ++n) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb;
        wmma::load_matrix_sync(vb, Vs + k * 16 * KLD + n * 16, KLD);
        wmma::mma_sync(oacc[n], pa, vb, oacc[n]);
      }
    }
    if (j0 + BN < jend) put(buf ^ 1);
    __syncthreads();                                    // tile j+1 visible; nobody still reads buffer `buf`
    buf ^= 1;
  }

  // ---- out = O / l, BN columns at a time through the scratch
  const float inv = 1.0f / l;
  half* orow = O + b * osb + hh * osh + (long)(q0 + r) * oss;
#pragma unroll
  for (int n = 0; n < D / 16; n += BN / 16) {
    __syncwarp();
#pragma unroll
    for (int t = 0; t < BN / 16; ++t)
      wmma::store_matrix_sync(scratch + t * 16, oacc[n + t], SLD, wmma::mem_row_major);
    __syncwarp();
    if (q0 + r < Sq)
      for (int c = 2 * h; c < BN; c += 4)
        *reinterpret_cast<half2*>(orow + n * 16 + c) = __floats2half2_rn(srow[c] * inv, srow[c + 1] * inv);
  }
}
}  // namespace

// q, k, v: (B, H, S, 128) fp16 views, any strides with a contiguous last dim (a (B, S, H, D) tensor transposed is fine)
// -> o with q's layout
// k / v may have fewer heads than q (GQA: q head h uses kv head h / (Hq / Hkv)). causal: bottom-right aligned (Sq <= Sk).
torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kHalf && k.scalar_type() == torch::kHalf &&
              v.scalar_type() == torch::kHalf, "fp16 CUDA tensors");
  const int D = q.size(3);
  TORCH_CHECK(q.dim() == 4 && (D == 128 || D == 256) && k.size(3) == D && v.size(3) == D, "head dim must be 128 or 256");
  TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1, "last dim contiguous");
  TORCH_CHECK(q.stride(2) % 8 == 0 && k.stride(2) % 8 == 0 && v.stride(2) % 8 == 0 &&
              q.stride(1) % 8 == 0 && k.stride(1) % 8 == 0 && v.stride(1) % 8 == 0, "16-byte aligned rows");
  TORCH_CHECK(k.sizes() == v.sizes() && q.size(0) == k.size(0) && k.size(1) > 0 && q.size(1) % k.size(1) == 0, "shapes (Hq a multiple of Hkv)");
  TORCH_CHECK(!causal || q.size(2) <= k.size(2), "causal needs Sq <= Sk");
  TORCH_CHECK(k.size(2) > 0, "no keys: softmax over an empty set");
  TORCH_CHECK((long)q.size(0) * q.size(1) <= 65535, "batch x heads > 65535 (grid.y limit)");
  for (const auto& t : {q, k, v})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 && t.stride(0) % 8 == 0,
                "16-byte aligned rows (base pointer and every stride)");
  const at::cuda::CUDAGuard guard(q.device());
  auto o = torch::empty_like(q);                 // same strides as q
  const int B = q.size(0), H = q.size(1), Sq = q.size(2), Sk = k.size(2);
  auto launch = [&](auto kern, size_t smem) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);   // per device; cheap
    dim3 grid((Sq + BM - 1) / BM, B * H);
    kern<<<grid, WARPS * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(q.data_ptr()), reinterpret_cast<const half*>(k.data_ptr()),
        reinterpret_cast<const half*>(v.data_ptr()), reinterpret_cast<half*>(o.data_ptr()), Sq, Sk,
        (float)(scale * 1.4426950408889634), H, H / (int)k.size(1), causal ? 1 : 0, q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2));
  };
  if (D == 128) launch(attn_fwd<128, 32>, smem_bytes<128, 32>());
  else          launch(attn_fwd<256, 16>, smem_bytes<256, 16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "V100 tensor-core attention forward (fp16, D=128/256, optional causal, GQA)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = false);
}
