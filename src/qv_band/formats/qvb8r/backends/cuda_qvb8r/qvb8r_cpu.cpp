// qvb8r CPU side: embedding rows (the table never touches VRAM, only the looked-up rows do) and the encoder.
//
// embed_rows: for each requested row, decode the tile holding it up to that row exactly as the GPU does -- per symbol
// pair, both symbols of every lane, then lanes in lane order take a 32-bit word when their 64-bit state fell below
// 2^32 (the order the GPU ballot hands words out). id = s + drow[row]; band value as QVB8's torch decode computes it
// (fp32 mul, then add); then the row's escapes overwrite their columns with the exact values.
// encode: the inverse walk (codec.encode_tensor), OpenMP over tiles.
#include <torch/extension.h>
#include <algorithm>
#include <cstdint>
#include <vector>

torch::Tensor embed_rows(torch::Tensor rows, torch::Tensor words, torch::Tensor off, torch::Tensor states,
    torch::Tensor tiles, torch::Tensor tab, torch::Tensor inv_perm, torch::Tensor row_tile, torch::Tensor mn,
    torch::Tensor ts, torch::Tensor drow, torch::Tensor esc_ptr, torch::Tensor esc_col, torch::Tensor esc_val, int64_t P) {
    for (const torch::Tensor& t : {rows, words, off, states, tiles, tab, inv_perm, row_tile, mn, ts, drow, esc_ptr, esc_col, esc_val})
        TORCH_CHECK(t.device().is_cpu() && t.is_contiguous(), "qvb8r embed_rows: inputs must be contiguous CPU tensors");
    TORCH_CHECK(P % 128 == 0, "qvb8r embed_rows: P % 128 == 0");
    const int64_t n = rows.numel(), Q = mn.numel(); const int S = (int)(P / 32);
    auto out = torch::empty({n, P}, torch::kFloat);
    const int64_t* R = rows.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) TORCH_CHECK(R[i] >= 0 && R[i] < Q, "qvb8r embed_rows: row id out of range");
    const uint32_t* Wd = (const uint32_t*)words.data_ptr<int32_t>();
    const int64_t* O = off.data_ptr<int64_t>(); const uint64_t* St = (const uint64_t*)states.data_ptr<int64_t>();
    const int64_t* Ti = tiles.data_ptr<int64_t>(); const uint32_t* Tb = (const uint32_t*)tab.data_ptr<int32_t>();
    const int32_t* IP = inv_perm.data_ptr<int32_t>(); const int32_t* RT = row_tile.data_ptr<int32_t>();
    const float* MN = mn.data_ptr<float>(); const float* TS = ts.data_ptr<float>(); const int32_t* DR = drow.data_ptr<int32_t>();
    const int64_t* EP = esc_ptr.data_ptr<int64_t>(); const int32_t* EC = esc_col.data_ptr<int32_t>();
    const float* EV = esc_val.data_ptr<float>();
    float* Y = out.data_ptr<float>();
    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < n; ++i) {
        const int64_t r = R[i]; const int sp = IP[r]; const int tile = RT[sp];
        const int k = sp - (int)Ti[3 * tile + 1]; const uint32_t* tb = Tb + Ti[3 * tile] * 4096;
        uint64_t x[32]; for (int j = 0; j < 32; ++j) x[j] = St[(int64_t)tile * 32 + j];
        int64_t ptr = O[tile]; float* y = Y + i * P;
        const float m = MN[r], t = TS[r]; const int dr = DR[r];
        for (int rr = 0; rr <= k; ++rr)
            for (int s0 = 0; s0 < S; s0 += 2)
                for (int j = 0; j < 32; ++j) {
                    for (int h = 0; h < 2; ++h) {
                        const uint32_t E = tb[x[j] & 4095u], s = E & 2047u;
                        x[j] = (uint64_t)((E >> 22) + 1u) * (x[j] >> 12) + ((E >> 11) & 2047u);
                        if (rr == k) { volatile float p = (float)((int)s + dr) * t; y[(s0 + h) * 32 + j] = m + p; }
                    }
                    if (x[j] < (1ull << 32)) x[j] = (x[j] << 32) | Wd[ptr++];
                }
        for (int64_t e = EP[r]; e < EP[r + 1]; ++e) y[EC[e]] = EV[e];
    }
    return out;
}

// sym: (R, P) int16 symbols, rows in sorted (tile) order. tiles: (nt, 3) int64 (class, first row, rows).
// F, C: (K, N) int64 quantised frequencies (sum 4096) and exclusive cumsums.
// -> words (u32 as i32), tile_off i64 [nt + 1], states (u64 as i64) [nt, 32].
// State in [2^32, 2^64). Per pair (decoder order s1, s2): if x >= 2^40 * f1 * f2 the low 32 bits are emitted first
// (x >>= 32), then s2 and s1 are encoded -- so after decoding both the state is >= 2^32 unless a word was emitted.
std::vector<torch::Tensor> encode(torch::Tensor sym, torch::Tensor tiles, torch::Tensor F, torch::Tensor C) {
    TORCH_CHECK(sym.dtype() == torch::kInt16 && sym.is_contiguous() && sym.dim() == 2 && sym.size(1) % 128 == 0,
                "qvb8r encode: sym must be contiguous (R, P) int16, P % 128 == 0");
    TORCH_CHECK(tiles.dtype() == torch::kInt64 && F.dtype() == torch::kInt64 && C.dtype() == torch::kInt64 &&
                tiles.is_contiguous() && F.is_contiguous() && C.is_contiguous(), "qvb8r encode: int64 tables");
    const int64_t P = sym.size(1), S = P / 32, nt = tiles.size(0), N = F.size(1);
    const int16_t* sp = sym.data_ptr<int16_t>(); const int64_t* tp = tiles.data_ptr<int64_t>();
    const int64_t* fp = F.data_ptr<int64_t>(); const int64_t* cp = C.data_ptr<int64_t>();
    auto states = torch::empty({nt, 32}, torch::kInt64);
    uint64_t* stp = reinterpret_cast<uint64_t*>(states.data_ptr<int64_t>());
    std::vector<std::vector<uint32_t>> out(nt);
    #pragma omp parallel for schedule(dynamic, 16)
    for (int64_t t = 0; t < nt; ++t) {
        const int64_t k = tp[3 * t], r0 = tp[3 * t + 1], nr = tp[3 * t + 2];
        const int64_t* f = fp + k * N; const int64_t* c = cp + k * N;
        uint64_t x[32];
        for (int l = 0; l < 32; ++l) x[l] = 1ull << 32;
        std::vector<uint32_t>& buf = out[t];
        buf.reserve((size_t)(nr * P / 6));
        for (int64_t p = nr * S / 2 - 1; p >= 0; --p) {
            const int64_t a = 2 * p, b = a + 1;
            const int16_t* s1p = sp + (r0 + a / S) * P + (a % S) * 32;
            const int16_t* s2p = sp + (r0 + b / S) * P + (b % S) * 32;
            for (int l = 31; l >= 0; --l) {                 // reversed; the whole buffer is reversed at the end
                const uint64_t f1 = f[s1p[l]], c1 = c[s1p[l]], f2 = f[s2p[l]], c2 = c[s2p[l]];
                uint64_t v = x[l];
                if (v >= ((f1 * f2) << 40)) { buf.push_back((uint32_t)v); v >>= 32; }
                v = ((v / f2) << 12) + v % f2 + c2;
                v = ((v / f1) << 12) + v % f1 + c1;
                x[l] = v;
            }
        }
        std::reverse(buf.begin(), buf.end());
        for (int l = 0; l < 32; ++l) stp[t * 32 + l] = x[l];
    }
    auto off = torch::empty({nt + 1}, torch::kInt64); int64_t* op = off.data_ptr<int64_t>(); op[0] = 0;
    for (int64_t t = 0; t < nt; ++t) op[t + 1] = op[t] + (int64_t)out[t].size();
    auto words = torch::empty({op[nt]}, torch::kInt32);
    uint32_t* wp = reinterpret_cast<uint32_t*>(words.data_ptr<int32_t>());
    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t t = 0; t < nt; ++t) std::copy(out[t].begin(), out[t].end(), wp + op[t]);
    return {words, off, states};
}

// per-row histograms (R, N) int32 of int16 symbols in [0, N)
torch::Tensor row_hist(torch::Tensor sym, int64_t N) {
    TORCH_CHECK(sym.dtype() == torch::kInt16 && sym.is_contiguous() && sym.dim() == 2, "qvb8r row_hist: (R, P) int16");
    const int64_t R = sym.size(0), P = sym.size(1); const int16_t* sp = sym.data_ptr<int16_t>();
    auto H = torch::zeros({R, N}, torch::kInt32); int32_t* hp = H.data_ptr<int32_t>();
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < R; ++r) for (int64_t j = 0; j < P; ++j) ++hp[r * N + sp[r * P + j]];
    return H;
}
