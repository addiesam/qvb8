# QVB8 Inference Engine: design

How the engine works and how each serving mode is built and set. The [README](README.md) says what it can do; this file says how.
Everything here is taken from the code in `src/`. A mode that is not tested is marked **not tested**.

Contents: [1. Architecture](#1-architecture) · [2. Key capabilities, in detail](#2-key-capabilities-in-detail) · [3. Serving modes](#3-serving-modes-what-each-one-does-and-how-to-set-it) · [4. Choosing a mode](#4-choosing-a-mode) · [5. Reference: requirements, install, options, speed](#5-reference)

---

## 1. Architecture

```
 client (Cline, curl, any OpenAI client)
        |  HTTP /v1/chat/completions   (streaming, tool calls, <think>)
        v
 +-------------------------------------------------------------+
 |  router  (optional)  one URL over many servers              |
 |  scores each backend: load, saved prefixes, link speed      |
 +--------+------------------------+---------------------------+
          |                        |
          v                        v
 +------------------+     +------------------+
 | LLM server       | ... | LLM server       |    one process per card, or one process over many cards
 |  admission queue |     |                  |
 |  continuous      |     |                  |
 |  batching engine |     |                  |
 |  speculative     |     |                  |
 |  decoding plan   |     |                  |
 |  CUDA-graph step |     |                  |
 +--------+---------+     +------------------+
          |
          v
 +-------------------------------------------------------------+
 |  model adapter -> target (a model's own forward)            |
 |    weights : QVB8 / QVB8R packs, read as packed ids         |
 |    KV cache: banded (certified) cache, fused attention      |
 |    state   : recurrent (Gated-DeltaNet) state, certified    |
 +-------------------------------------------------------------+
          |
   VRAM  <-->  host RAM (pinned, /dev/shm)  <-->  another machine (DRAM PC, KV host, stage worker)
```

| Layer | What it does | Code |
|---|---|---|
| Pack reader | reads a weight as packed ids plus per-row `(min, tau)`; the GPU restores `min + id * tau_s` in the kernel | `qv_band/formats/qvb8`, `qvb8r` |
| Adapter / target | builds one model's forward and exposes the common `BatchTarget` surface | `qv_band/engine/adapter.py`, `models/` |
| Batching engine | continuous batching: prompt chunks and one-token decodes in one batch, prompt-prefix reuse | `engine/target_engine.py` |
| Speculative layer | decides how many drafts to verify each step | `engine/spec.py` |
| Static decode | one token of every live request as one CUDA-graph replay | `engine/static_decode.py` |
| Server | OpenAI API, admission control (`--max-live`, `--max-queue`: full queue gives 429, long wait gives 503) | `ai_engine/servers/llm_server.py` |
| Router | one endpoint over LLM servers and job servers | `ai_engine/servers/router.py` |

---

## 2. Key capabilities, in detail

### 2.1 A full serving engine, not just a format

**One speculative-decoding layer for every model.** A model that drafts for itself (a DFlash / DFlash2 / DSpine drafter, or the DSpark drafter inside Inkling's checkpoint) implements three calls:
`spec_propose` (draft k tokens), `spec_verify` (run the target over `[last] + drafts`, return the logits of every token) and `spec_keep` (keep the accepted prefix, roll the rest back).
The same plan serves one card, several cards in one process, one process per card, tensor-parallel groups and image prompts.

```
 step n:    last token  ->  drafter  ->  d1 d2 d3 d4 d5 d6 d7        (draft k tokens)
            target verifies [last d1 ... d7] in ONE forward
            accept while the target agrees with the draft:     d1 d2 d3 (x)  d4 rejected
            emit d1 d2 d3 + the target's own token at the first mismatch
 step n+1:  continue from there; the drafter follows the accepted prefix
```

*Which drafts are verified* is not fixed: draft j of a row survives with the product of the drafter's first j probabilities, and the verify rows go to the most likely drafts of any live request
while they still raise the expected tokens per second, using the card's **measured** step cost. Text the drafter predicts badly therefore costs no extra time.
Acceptance follows each request's sampling settings. At temperature 0 it is greedy: a draft is kept while it equals the target's own token, so the emitted text is the target's own greedy text. Above 0 it is speculative sampling: each draft is kept with probability min(1, p_target / p_draft) and a rejection draws from the residual distribution, so the output follows the target's own sampling distribution (temperature, top_k, top_p). One implementation (`engine/spec.py`) serves every engine. Tensor-parallel cards other than the first only verify; they run no drafter, and a sampled decision is made on the first card and broadcast, so the cards stay in step. A drafter that samples (Inkling-2-Mini's DSpark) draws its drafts from its own filtered distribution and hands it to the acceptance step, so sampled requests speculate as well.

**CUDA-graph decoding.** A model's per-request state normally grows (`torch.cat`), so a decode step is thousands of tiny kernel launches and the GPU waits for Python.
Here the state of every decoding request lives in fixed buffers, one row per slot, updated in place with positions in device tensors, so one step of all live requests is one graph replay.
The slots are sized at start from a measured prefill, so a long prompt still fits; a request whose context would pass the slot length stays on the eager path.

**OpenAI-compatible API.** `/v1/chat/completions` with streaming, JSON tool calls, the `<think>` channel and content-part lists (as sent by Cline). Several servers sit behind one router URL.

### 2.2 Zero-expansion operating VRAM

A weight is never expanded to bf16 in memory. A QVB8 row stores integer band ids at the row's own bit width plus `(min_s, tau_s)` once per row; the matmul kernel restores `min_s + id * tau_s` in fp32 inside the kernel, one fused multiply-add.
So the card holds the pack at its packed size, and only a tile is restored in registers at a time.

```
 bf16 weight   :  [ w w w w w w w w ]   16 bits each     54 GB for Qwen3.8-27B
 QVB8 row      :  [ i i i i i i i i ]   ~5-8 bits each   + (min_s, tau_s) per row
                         |
                         v   inside the kernel:  w^ = min_s + i * tau_s   (one FMA, fp32)
 QVB8R         :  the same ids, rANS-coded in row tiles (smaller, 13.6 GB at tau 2e-3), decoded on the GPU
```

QVB8R is smaller than QVB8 (same ids, entropy coded). Cutting a QVB8R table for tensor parallel is done in place from the coded stream (decoder states at the cut, no re-encoding); when a cut does not fit the tile grid a re-encode fallback is used.

### 2.3 Certified worst-case accuracy

Every restored weight satisfies `|w^ - w| <= tau`, checked after packing, element by element, and the largest error is recorded. A weight that cannot be packed smaller than its source at its tau stays dense.

- **No codebook:** each row is banded against a *virtual* codebook `c_i = mu_r + i * tau_s` with `tau_s = 2 tau`, so a weight is restored from its index alone.
- **The certificate is the number the GPU writes:** the kernel's `fmaf` was compared bit for bit with a true single-rounding fused multiply-add (Decimal arithmetic) over 2x10^5 trials.
- **Decode arithmetic stays in fp32:** computing `i * tau_s` in a narrow format can exceed tau by up to 1.95 tau once `i` passes 2^10, so only the final value is narrowed.
- **A floor:** any certified, randomly accessible representation needs at least `ceil(log2 N_tau)` bits per element (N_tau = the tau-covering number of the weights).
- **Both dtypes:** a pack is certified for fp16 and bf16; fp16 needs `escape_tables.json`, the few weights an fp16 decode must take from the stored originals.

The bound covers the compressed weights (and the KV cache and recurrent state, 2.6). It does not promise that answers are correct or identical to another implementation.

#### One weight form for every source format

The engine reads every weight in one form. For a weight `w` in band `b` with integer index `q`:

```
w = m_b + s_b · T(q)
```

A **band** is the group of weights that shares one offset `m_b` and one step `s_b`; `T` maps the index to a value. QVB8 is the case of a uniform `T(q) = q` with one band per row. The low-precision formats that models are published in fit the same form with `m_b = 0`, another band shape, and either a uniform `T` or the value table of a floating-point code.

| Format | Band | T(q) | m_b | s_b |
|---|---|---|---|---|
| QVB8 / QVB8R | one row | `q` (uniform) | row minimum | `2τ` (certified) |
| Int8 / int4 group (GGUF Q8_0 / Q4_0 convention) | 32 weights | `q` (uniform, signed) | 0 | absmax / 127 (Q8_0), absmax / 8 (Q4_0) |
| NVFP4 | 16 weights | E2M1 value table (16 values) | 0 | E4M3 block scale × per-tensor fp32 scale |
| MXFP8 | 32 weights | E4M3 value table | 0 | E8M0 scale (2^(e−127)) |
| Block FP8 | 128 × 128 block | E4M3 value table | 0 | fp32 block scale |

**Implemented:** QVB8 / QVB8R, and NVFP4, MXFP8 and block FP8 as one format, `qvbq`. The int8 / int4 group row shows the form only: the engine does not load GGUF files.

**How `qvbq` is served.** The checkpoint's own bytes are kept and each weight is decoded inside the matrix multiplication, `w[e, n, k] = CODE(q) · SCALE(s[e, n // GN, k // GK]) · f[e, n]`: `CODE` is E2M1, E4M3 or E5M2; `SCALE` an E4M3 byte (NVFP4, per 16), an E8M0 byte (MX, per 32) or fp32 (FP8 128 × 128, per channel or per tensor); `f` an optional fp32 per-row factor that holds NVFP4's per-tensor scale, so the gate and up rows of an expert can be concatenated without changing any value. The restored value equals the published value: the error against the quantised checkpoint is 0.

**What this gives.** One loader, one kernel family and one saved-state format serve every format, and a model can mix formats tensor by tensor. Inkling-2-Mini is served this way: its routed experts stay in their published NVFP4 / MXFP8 form, and the other matrices are QVB8 packs (τ = 1e-3). Code: `src/qv_band/formats/qvbq/` (`kernel.py` the on-the-fly matmul, `module.py` the linear / embedding / expert modules, including experts in host RAM). Paper: *Applying Certified L∞-Bounded Compression to LLM Serving*, Section 6, Table 4.

### 2.4 One card or mixed clusters

The code is not tied to a GPU generation: kernels exist for sm_70 (V100: register-blocked GEMV, WMMA attention, no bf16, no flash-attn) and sm_121a (GB10), and compile on first use for other cards.
What moves between machines is **data defined by numbers** (integer ids plus per-channel min and step), not by a kernel, so a prefix computed on a GB10 restores on a V100 with the same answer (section 3.10).
Cards of different sizes work in one pipeline: layers are placed by weight size and each card's free memory (`layer_plan.py`). Tested on a DGX Spark (GB10) and V100 cards only.

### 2.5 Full-pipeline compression

One packer and one loader (`run/weight_load.py`) serve every component: LLM, DiT (transformer), text encoder, VAE, UNet, audio, LoRA, DFlash drafter, embedding and reranker.
A weight goes whole onto a card, as a tensor-parallel share, into host RAM, or stays in its file; how it is loaded does not change. This is why a whole pipeline (script, image, video, voice) fits on one small machine.

### 2.6 A KV cache that is certified, small, on-the-fly and shareable

```
 new tokens --> [ residual buffer: last 256 tokens, dense ] --(full)--> band in bulk --> packed ids
                                                                                         |
 attention: q . k_t = sum_c (q_c * step_c) * id[t,c]  +  q . min        <-- reads the ids directly
```

- **Certified:** keys and values are coded per channel as `min[c] + id * step[c]`; the same bound as the weights. The recurrent linear-attention state (Gated-DeltaNet) is certified the same way. No token is dropped by eviction.
- **Small:** about 26.5 KB per token on Qwen3.8-27B at KV tau 0.2, against 64 KB in 16-bit (150k tokens: 4.1 GB instead of 9.8 GB, +0.006% perplexity).
- **On the fly:** the newest tokens sit dense in a small residual buffer (`--kv-residual`, 256) and are banded in bulk once it is full. Attention (`kv/fused_attn.py`) reads the packed ids directly: the dequantisation is folded into the query side, so there is no dense copy and no bf16 shadow buffer.
- **Shareable:** the saved state is plain numbers, so it can sit in host RAM, on a DRAM PC, or move to another machine (sections 3.7, 3.9 and 3.10).

### 2.7 Built for production workflows

An agent (Cline, OpenCode) sends the whole history again every round. The engine stores the state at the end of every prompt and at its last whole-block boundary (LRU, byte budget); a later prompt that extends a stored one token for token prefills only the new text, and an identical resend samples from stored logits with no forward.
Older history moves to host RAM (`--kv-window`) or to another machine, so a long session keeps going. Admission control keeps a small card from being overloaded (`MAX_LIVE=1` on a 16 GB card).
For media, a router with job servers puts script, image, video and voice stages behind one URL (image, video and audio models are not released yet).

---

## 3. Serving modes: what each one does and how to set it

Modes combine. Each is an independent choice of **where the weights live**, **how work is split**, and **where the KV and saved prefixes live**.

### 3.1 Single card

```
   request --> [ GPU 0:  all layers + KV ] --> answer
```

One process, one card, the whole pack in VRAM. Set: the launcher with no parallel flag.
`scripts/qwen_server.sh start`  /  16 GB card, 27B: the 13.6 GB QVB8R pack plus DRAM offload (3.7).
**Best for** a model that fits. **Limit:** one request in the live batch on a 16 GB card.

### 3.2 PP: pipeline parallel (layers split over cards, one process)

```
   request --> [ GPU 0: layers 0..a ] --h--> [ GPU 1: layers a+1..b ] --h--> ... --> [ GPU N: last layers
                                                                                          + embedding, norm, lm_head,
                                                                                            drafter, request tensors ] --> answer
```

Each card holds a **contiguous range** of decoder layers; a layer's inputs are moved to its card and that card becomes current while the layer runs, so the model's forward is device-transparent.
**The last card is the main card.** Without a split given, the split balances weight bytes (the main card counts its non-layer weights too): same pack and same cards give the same split, printed at load.

Set: `--pipeline cuda:0,cuda:1` (cards in layer order) or `--pipeline auto` (every visible card); by hand `--pipeline-split s1,...` (first layer of cards 2..N); `--pipeline-reserve-mb` (default 2048) keeps memory free for activations and KV.
Example: `scripts/qwen_server.sh start --pipeline auto`.

**What it buys:** memory (a pack larger than one card loads). **What it does not buy:** speed for one message, because a token passes the layers in order. Measured on 2 x V100 (Qwen3.8-27B, about 110 tokens): single 16 s, PP 13 s.

### 3.3 DP: data parallel (a replica per card, one router)

```
                         +--> [ server A : GPU 0, whole model ] --+
   client --> router --> |                                        +--> answers
                         +--> [ server B : GPU 1, whole model ] --+
```

Each card runs the **whole model** in its own server (own port); the router chooses a backend for every request. For an LLM it scores backends by load and by **which one already holds the request's saved prefix**, and a prefix is copied to another backend only when copying is faster than recomputing (measured link speed).
Several servers on one machine share one RAM pool of prefixes with `--prefix-store shared` (3.9), so a document prefilled on card 0 is restored on card 1 and stored once.

Set: start the same launcher on each card (own `PORT`), then
```bash
python -m ai_engine.servers.router --port 8000 --backend a=http://localhost:8081 --backend b=http://localhost:8082
```
**What it buys:** requests per second (many users), not speed per message. **Limit:** each card must hold the whole model.

### 3.4 TP: tensor parallel (every layer split over cards)

```
                 token t (the SAME token on every card)
           +--------------------+--------------------+
           v                    v                    v
      [ GPU 0 ]            [ GPU 1 ]            ...
      heads 0..h/2         heads h/2..h               column-parallel: q, k, v, MLP up
      MLP part 0           MLP part 1                 (split by OUTPUT rows, no communication)
           |                    |
           +----- all-reduce ---+                     row-parallel: o_proj, MLP down
                      |                               (split by INPUT columns: a partial sum each)
                      v                               ONE all-reduce per block half
                 same logits on every card; card 0 answers
```

Every card works on the same token at the same time. A block's first projections are split by output rows (each card owns its heads or its part of the MLP); its last projection is split by input columns, then one all-reduce. Both splits are **exact for QVB8** because the certificate is per weight: a row carries its own `(min_s, tau_s, bit width)`, and a column range of a row is a fixed-width bit range, so every restored weight is still within tau.
For mixture-of-experts models the experts are split too (**expert parallel**): card r owns whole experts, every card routes all tokens, runs the pairs for its experts, and one reduce sums routed experts plus the shared branch.

Two implementations (a model adapter opts in):

| | threads (one process) | processes (`torchrun`, NCCL) |
|---|---|---|
| Set | `QV_TP_DEVICES=cuda:0,cuda:1` | `torchrun --nproc-per-node N -m ai_engine.cli.serve ...` (rank 0 serves HTTP) |
| All-reduce | thread barrier; cards share one GIL | NCCL, **inside the CUDA graph** |
| Speculative decoding | card 0 drafts, all cards verify | rank 0 drafts, followers only verify |

Pre-cut shares: `QV_TP_SHARD_WRITE=DIR` saves every share a process cuts, `QV_TP_SHARD_DIR=DIR` reads them back (a share whose source file changed stops the load with the reason).
A TP prefix restores only into the **same split** (`{"tp": [card 0, card 1, ...]}`).

**What it buys:** one message gets faster. Measured on 2 x V100, no P2P, Qwen3.8-27B (~110 tokens): single 16 s, PP 13 s, **TP 7 s**. Card-to-card copies go through host memory on that board, so TP did not beat one card for a model that fits one card (Z-Image).
**Qwen-family TP server** (`ai_engine/servers/tp_server.py`, launcher `scripts/qwen_tp_server.sh`): one process per card under `torchrun` (NCCL). Rank 0 runs HTTP and the queue and broadcasts each request to every card; decode runs as CUDA graphs with DFlash and exact verification. Two serving shapes, set at start: the default is the single-message speed path (one conversation, DFlash drafter, requests wait in arrival order); `--batch B` (launcher: `USERS=B`) decodes up to **B conversations together**, each in its own slot with its own context of `--max-len` (a drafter is ignored unless `QV_BATCH_DRAFT=1`, because 4 users with a drafter measured 16.7 tok/s in total against 34 for the plain batch). Each request's temperature, top_k and top_p apply in both (temperature 0: greedy); with a drafter, sampled requests use speculative sampling, and the sampled decision is made on card 0 and broadcast to the other cards. Measured on 2 x V100, Qwen3.8-27B: one request 11.5 tok/s, four at once 38.6 tok/s in total (3.4x); the single-user DFlash path runs 27.5 to 35 tok/s. To the router it is one replica (`GET /v1/pool/state`). A model adapter can also opt in to TP without this server (Inkling-2-Mini: `QV_TP_DEVICES=cuda:0,cuda:1 scripts/inkling_server.sh start`).

```bash
MODEL=./models/Qwen3.8-27B_qvb8_1e-3_fp16 DRAFT=./models/Qwen3.8-27B-DFlash2_qvb8_2e-3_fp16 scripts/qwen_tp_server.sh start
```

### 3.5 TP + DP (groups of TP cards behind a router)

```
                          +--> [ TP group 1: GPU 0 + GPU 1 ] --+
   client --> router -->  |                                     +--> answers
                          +--> [ TP group 2: GPU 2 + GPU 3 ] --+
```

Each router backend is itself a TP server over its own cards: the router sees a TP server as **one replica**, so TP makes a message fast and DP adds users. Set: start one TP server per card pair with `CUDA_VISIBLE_DEVICES=0,1` and `2,3` on different ports, then the router as in 3.3. Not yet run on four cards (the test box has two); the pieces (TP server, router) are each tested alone.

### 3.6 PP over machines (one model, several machines)

```
   machine B (stage worker)                     machine A (main)
   [ layers 0..11 ] --h (TCP, one connection)--> [ layers 12..L + embedding, lm_head, drafter, scheduler ] --> answer
```

A stage worker runs layers `[a, b)` and keeps their cache; the main process runs the engine and replaces layer a's forward by **one round trip** (hidden states out, hidden states after layer b-1 back). Every cache action of the engine on a slot is repeated on the stage.
Transport (`stage_link.py`): one persistent TCP connection per neighbour; message = a small JSON header plus raw tensor bytes (nothing is pickled). A dead neighbour is detected in about a minute (keep-alive probes after 30 s, 3 unanswered = lost). Any error stops the requests in flight; there is no reconnect.

Set:
```bash
# on the other machine: runs layers 0 to 12
python -m ai_engine.cli.stage_worker $MODEL --stage-port 8081 --stage-layers 0:12
# on this machine: the rest
scripts/qwen_server.sh start --pipeline host:8081,local --pipeline-split 12
```
Set the same `QV_POOL_TOKEN` on both (without it the worker port is open to its network). The worker loads only its layers (`QV_LAYERS_HERE=a:b,only`); the other weights are never read. Tested: two V100 cards and a GB10 over Wi-Fi.

### 3.7 DRAM: older KV, saved prefixes and the vision tower in host RAM

```
   VRAM:  [ weights ][ newest KV (window) ][ working buffers ]
   RAM :  [ older KV blocks ][ saved prefixes ][ vision tower ]      pinned host memory
```

Set: `--kv-window N|auto`, `--kv-cold-block T`, `--kv-gpu-reserve-mb M`, `--prefix-store host`, `--vision-tower host`.
`--kv-window N` keeps the newest N tokens per layer in VRAM; `auto` keeps everything in VRAM while there is room and, once free VRAM would drop below the reserve, freezes the window and moves only the oldest blocks. Older blocks are read back in cold blocks (`--kv-cold-block`) and merged by log-sum-exp.
Example (27B on one 16 GB card, 15k-token prompt, measured on a V100): `--prefill-chunk 512 --kv-window 4096 --kv-cold-block 8192 --prefix-store host --vision-tower host`: 4.9 tok/s decode, 14.8 GB of 16 used.
A 94k-token prompt pins about 9.5 GB of host RAM.

### 3.8 DRAM + disk: where the weights come from (weight tiers)

```
   tier      weights stream from                          RAM used
   vram      the card (only what does not fit streams)    none
   dram      pinned host RAM                              all streamed weights
   cache     the files; pages stay in the page cache      reclaimable by the kernel
   disk      the files; pages dropped after each read     least
   share:    shared = ONE host copy in /dev/shm for every server of the machine; own = one copy per server
```

Set: `QV_PLACE="transformer=dram:shared,text_encoder=cache:shared,vae=vram"` (per sub-model; `*` = the rest), or the launcher's `QV_STREAM_FROM` and `QV_STREAM_HOST_GB`. Nothing here writes to a disk. A second server attaches to the owner's `/dev/shm` buffers instead of reading the file.
For an **LLM expert model** that does not fit: `QV_EXPERTS=host` keeps the experts in host RAM with `QV_EXPERT_CACHE_GB` of them on the card.

**Block streaming** (`swap/block_stream.py`) is for image and video transformers: a window of `ahead + 1` blocks is on the card, the next blocks are copied over a second stream while the current one computes, and the result is bit-identical to a resident run. It hides the copy because every weight is used for thousands of tokens per pass; it is **not** for an LLM decoding one to eight tokens per pass (that is bound by exactly this copy).

### 3.9 DRAM PC as a pool, and a KV host

```
   GPU machine A ----+                      +--  prefix store (plain Python, no GPU):
   GPU machine B ----+---- LAN ---->  DRAM PC     mirrors every saved prefix, blake2b-checked
   GPU machine C ----+                      +--  KV host: holds OLD KV blocks and computes attention over them
```

- **Prefix store** (`ai_engine/servers/prefix_store.py`, standard library only): `python prefix_store.py --mirror http://host-b:8080 --budget-gb 6 --port 8082`. It polls each source and copies every prefix it does not hold; the router lists it with `--store pc=http://...` as a donor. Servers use `--prefix-backup http://pc:8082`, so their own pool can stay small.
- **KV host** (`ai_engine/servers/kv_host.py`): `python kv_host.py --port 8083 --budget-gb 8`. A server whose memory is full sets `--kv-remote http://pc:8083`; it ships each layer's oldest KV blocks there and, per layer and step, sends only the **queries**; the host returns `(o, lse)` over the blocks it holds and the main machine merges by log-sum-exp. Old KV is read at every step, so shipping it back each step would cost more than the attention.
- **Shared pool on one machine** (`--prefix-store shared`): a POSIX shared-memory arena in `/dev/shm` (RAM, never disk), registered with CUDA in each process. A prefix is stored once; readers get views and hold a lease until their request leaves.

### 3.10 Many machines, different cards (replicas + shared state)

```
   GB10 (Blackwell) ---+                    prefix computed on the GB10
   V100 box (Volta) ---+---- router ----    is COPIED to a V100 (hash-checked), restored, same answer
   DRAM PC ------------+
```

Every machine runs the whole model; one router URL in front; saved prefixes move between machines. The saved state is certified integer ids plus per-channel `(min, step)`; it is defined by the numbers, so any GPU can restore it.
A copy needs **the same transfer signature on both sides** (pack and tau, state tau, KV tau, KV residual, checkpoint grid, dtype, drafter) before one byte moves. The receiver publishes a prefix only after every piece arrived with a matching blake2b hash; a failed pull costs only the prefill it would have saved.

Set: each machine `scripts/qwen_server.sh start` (own port), then
```bash
python -m ai_engine.servers.router --port 8000 \
    --backend gb10=http://gb10:8080 --backend v100a=http://v100host:8080 --backend v100b=http://v100host:8081
```
Set the same `QV_POOL_TOKEN` on every server, the router and the store: export, piece, release and pull need it; without it they are open to the LAN.
Measured (Qwen3.8-27B, a 15,316-token prompt, GB10 to V100, Wi-Fi test link): V100 cold 40.2 s; after pulling the GB10's 12,288-token prefix, 10.3 s with an identical answer. Tested on these two GPU types only.

### 3.11 Job servers (image, video, audio) behind the router

The same router fronts job servers. A job route goes to the backend with the smallest wait, `(queued + running jobs) x last job duration`; the job id is remembered, so `GET /v1/jobs/<id>`, cancel and result files reach the backend that holds the job. Multi-stage pipelines (script, image, video, voice) are several servers behind one URL.
Image, video and audio model packs are not released yet.

---

## 4. Choosing a mode

| Goal | Mode | Set with | Measured / status |
|---|---|---|---|
| Model fits one card | single | launcher | |
| Model too big for one card | PP | `--pipeline auto` | tested, 2 x V100 and GB10 + V100 |
| Faster single message | TP | `scripts/qwen_tp_server.sh`, or `QV_TP_DEVICES` (Inkling) | 27B on 2 x V100: 7 s vs 16 s; `scripts/qwen_tp_server.sh` |
| More users at once | DP | one server per card + router | tested |
| Fast messages and many users | TP + DP | TP servers behind a router | not yet run on 4 cards |
| Long context on a small card | DRAM | `--kv-window`, `--prefix-store host` | tested, V100 16 GB |
| Model bigger than VRAM | DRAM + disk | `QV_PLACE`, `QV_EXPERTS=host` | tested for DiT and MoE |
| Many machines, one model | PP over machines | `stage_worker` + `--pipeline host:port,local` | tested, 2 x V100 + GB10 over Wi-Fi |
| Many machines, many users | DP over machines | router across machines | tested, GB10 + V100 |
| Share saved state | DRAM PC / KV host | `prefix_store.py`, `kv_host.py` | tested |

Measured on two V100 16 GB (no P2P), one request each: Qwen3.8-27B, about 110 tokens: single 16 s, PP 13 s, TP 7 s, DP 14-45 s. A model that cannot fit is **refused at start** with the numbers (for example Qwen3.6-35B-A3B: needs 29.7 GiB, the cards have 2 x 13), instead of failing later.

---

## 5. Reference

### Requirements

- Linux, an NVIDIA GPU and its driver. Python 3.10+ (tested 3.12) and the packages in `requirements.txt` (tested versions pinned).
- **Tested build:** NVIDIA DGX Spark (GB10, sm_121, aarch64), Ubuntu 24.04, driver 580, CUDA 13.0, PyTorch 2.14. Its prebuilt CUDA kernels are in
  `src/qv_band/.prebuilt/sm_121a/`: no CUDA toolkit needed.
- **V100 (sm_70):** prebuilt kernels in `src/qv_band/.prebuilt/sm_70/` (PyTorch 2.14.1, CUDA 12.6), also no toolkit needed. CUDA 13 dropped sm_70, so a V100 needs CUDA 12.x.
- **How a bundled kernel is trusted:** each `.so` has a `.json` sidecar whose key is a hash of its sources, flags, arch and PyTorch version. The loader uses the bundled kernel only when the key matches (and the platform), otherwise it compiles for your GPU (`nvcc` needed) into `.ext_cache/`. Bundled: `qvb8_cuda`, `qvb8r_cuda`, `qvattn_cuda` for sm_70 and sm_121a, and `qv_v100_attn` for sm_70.
- **Other GPUs:** the kernels compile for your GPU on first use (a few minutes, `nvcc` required) into `.ext_cache/` and are reused. Only the GB10 and V100 are tested.
- **Compute dtype (`--dtype bf16|fp16`):** a GPU without bf16 (V100) emulates it slowly: run fp16. On A100, H100, RTX 30/40/50 and DGX Spark both work; test which is faster.
  The packs are certified for both, so every restored weight is within τ in either. fp16 tops out at 65504, so an activation can overflow in fp16 where bf16 would not.
- **`escape_tables.json`** (needed for fp16): each pack folder carries it next to `manifest.json`. An fp16 run without it stops at start-up; a bf16 run does not need it.
- **flash-attn** is needed only by Ripple (its sliding-window layers). Other models run with `ATTN=sdpa`.

### Install

```bash
git clone https://github.com/addiesam/qvb8.git && cd qvb8
./install.sh        # creates .venv, installs PyTorch for your driver and the other packages, checks the GPU
```

Flash-attn for Ripple, built against your PyTorch: `pip install flash-attn==2.8.3.post1 --no-build-isolation`. With PyTorch 2.14 the sdist's `setup.py`
passes `-std=c++17`; replace it with `-std=c++20` before building. `FLASH_ATTN_CUDA_ARCHS` (for example `80` or `120`) limits the build to one GPU generation.

### Run a server

Each model family has a launcher; its header lists every setting. The default port is 8080 (`PORT=...`).

Qwen3.8-27B:

```bash
MODEL=./models/Qwen3.8-27B_qvb8r_2e-3_fp16 DRAFT=./models/Qwen3.8-27B-DFlash2_qvb8r_5e-3_fp16 DRAFT_K=7 scripts/qwen_server.sh start
```

Ripple-65B-Preview:

```bash
MODEL=./models/Ripple-65B-Preview-QVB8-1e-3 DRAFT=./models/Ripple-65B-Preview-DFlash2-QVB8-5e-3 scripts/ripple_server.sh start
```

SenseNova-Loop-9B:

```bash
MODEL=./models/SenseNova-Loop-9B_qvb8_5e-4 DRAFT=./models/SenseNova-Loop-9B-DFlash2_qvb8_5e-4 scripts/sensenova_server.sh start
```

Inkling-2-Mini:

```bash
MODEL=./models/Inkling-2-Mini_qvb8_1e-3 scripts/inkling_server.sh start
```

Every launcher takes `start | stop | status | restart | log | startup`, and extra flags after them go to the server and win (for example `restart --no-think`).

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a haiku about rivers"}],"max_tokens":200,"stream":false}'
```

The server speaks the OpenAI chat API: streaming, tool calls (JSON), the reasoning (`<think>`) channel and content-part lists as sent by Cline and similar clients.
`scripts/chat.sh` is a terminal chat client (`/reset` new conversation, `/exit` leave; `URL=http://host:8080` for a server on another machine).

#### Several cards and machines

- **One machine, several cards:** `--pipeline auto` (every visible card) or `--pipeline cuda:0,cuda:1` (in layer order; the last is the main card). Layers are placed by weight size
  and free memory, and the split is printed at load; `--pipeline-split` sets it by hand.
- **Several machines:** on the other box `python -m ai_engine.cli.stage_worker $MODEL --stage-port 8081 --stage-layers 0:12`, here
  `--pipeline host:8081,local --pipeline-split 12`. Set the same `QV_POOL_TOKEN` in both processes, otherwise the worker port is open to its network.
- **One URL over replicas:** `python -m ai_engine.servers.router --port 8000 --backend a=http://hostA:8080 --backend b=http://hostB:8080`.

### Supported options

GPU only; defaults are in the launcher.

- `--kv-cache qvb8` (certified compressed KV) or `bf16`; `--prefill-linear qvb8`.
- `--drafter dflash` (DFlash, DFlash2, DSpine, LiLiCorr: the kind is read from the drafter's config), `dspark`, `n-gram`, `n-gram_with_dflash`, or `--draft 0` for none.
- `--pipeline`, `--pipeline-split`, `--pipeline-reserve-mb` (default 2048 kept free per card).
- `--kv-window N|auto`, `--kv-host`, `--kv-cold-block T`, `--kv-gpu-reserve-mb M`: older KV in system RAM.
- `--prefill-chunk N`: a smaller chunk makes a repeated prompt or the next chat turn cheaper; a larger one prefills a new long document faster.
- `--prefix-store gpu|host|shared`, `--prefix-shared-name`, `--prefix-shared-gb`: where saved prefixes live; `shared` is one `/dev/shm` pool for every server of the machine.
- `--max-live`, `--max-live-tokens`, `--max-queue`, `--queue-wait`: admission control (0 = no limit). A full queue answers 429, a wait past the limit 503, both with `Retry-After`.

### KV cache shared between machines and different GPUs

A prompt prefilled on one machine is restored on another machine with a different GPU instead of being computed again.

The saved KV is certified data: integer ids plus a per-channel (min, step), value = min + id × step, with a proven error bound τ. It is defined by the numbers, not by a GPU or a kernel, so any machine can restore it.

- **Small and exact:** about 28 KB per token for Qwen3.8-27B, with the same error bound on whatever hardware restores it.
- **Copied, not recomputed:** the receiver publishes a prefix only after every piece arrived and its blake2b hash matched.
- **Safe by signature:** both sides must report the same transfer signature (pack and τ, KV τ, KV residual, checkpoint grid, dtype, drafter) before a byte moves.
- **Cost-aware router:** a prefix is pulled only when copying is faster than recomputing, from the measured link speed.
- **The store needs no GPU:** a 16 GB PC running plain Python holds the copies (`--prefix-backup`, `--kv-remote`).

Tested on two GPU types only (DGX Spark and V100). If you try yours, please report how it went in the GitHub issues.

**Measured (release v1.2.0, 2026-11-15):** Qwen3.8-27B (qvb8r, τ 2e-3, fp16), one prompt of 15,316 tokens, greedy, one run each; a GB10, one V100, a PC store and the router.
A functional check, not a benchmark; the network was a Wi-Fi test link.

| Step | Result |
|---|---|
| GB10 prefills the prompt (cold) | 19.8 s; the same prompt again, restored from its own pool: 4.9 s |
| V100, a different prompt of the same length (cold) | 40.2 s |
| V100 pulls the saved prefix (12,288 tokens, 496 MB) from the GB10 | 335 s at 1.5 MB/s on Wi-Fi (about 5 s at 1 Gb/s, calculated) |
| V100 answers the same prompt after the pull | restored 12,288 of 15,316 tokens, prefilled 3,028: 10.3 s, answer identical to the GB10's |
| Router, same prompt | sent to the V100 holding the prefix (14,336 tokens cached): 4.3 s |

### Speed

#### DGX Spark GB10 (measured 2026-11-16)

Release code and launcher as shipped, `ATTN=sdpa`, KV cache qvb8, speculative decoding, greedy, thinking off, 400 generated tokens, streaming; the second of two runs. One idle machine.

| Pair (target + drafter) | Decode, code | Decode, prose | Two requests at once (total) | Prefill (57k-token prompt) |
|---|---|---|---|---|
| Qwen3.8-27B qvb8 1e-3 + DFlash2 qvb8r 5e-3 (K=7), `--prefill-chunk 12288` | 18.7 tok/s | 14.4 tok/s | 23.9 tok/s | about 980 tok/s (58 s) |
| Qwen3.6-35B-A3B qvb8 5e-4 + DFlash2 qvb8 1e-3 (K=7) | 53.6 tok/s | 37.2 tok/s | 50.2 tok/s | about 1,920 tok/s (30 s) |
| Qwen3.8-27B qvb8r 2e-3 + DFlash2 qvb8r 5e-3 (K=7), the 16 GB-card pack | 15.5 tok/s | 12.0 tok/s | 18.9 tok/s | about 760 tok/s (75 s) |
| Qwen3.5-9B qvb8 1e-3 + DFlash qvb8 1e-3 (K=15) | 62.7 tok/s | 45.1 tok/s | 68.8 tok/s | about 2,700 tok/s (21 s) |

Chunk 12288 was fastest for the 27B on the GB10 (4096: about 880 tok/s, 24576: about 855 tok/s). The qvb8 pack is faster than qvb8r; qvb8r is smaller, which is what lets the 27B fit one 16 GB card.
Code decodes faster than prose because the drafter guesses code better. Speculative decoding is verified by the full model; in our tests greedy output matched plain decoding, and where two tokens are an exact near-tie the choice can differ.

#### V100 16 GB (measured 2026-11-16, idle machine)

Qwen3.8-27B qvb8r 2e-3 + DFlash2 qvb8r 5e-3 (K=7), `ATTN=sdpa`, one live request (`MAX_LIVE=1`), prefill with a 15,263-token prompt.

| V100 setup | Decode, code | Decode, prose | Prefill (15k-token prompt) | GPU memory |
|---|---|---|---|---|
| One card, DRAM offload (`--prefill-chunk 512 --kv-window 4096 --kv-cold-block 8192 --prefix-store host --vision-tower host`) | 4.9 tok/s | 3.1 tok/s | about 135 tok/s (113 s) | 14.8 GB of 16 |
| Two cards, `--pipeline auto --prefill-chunk 2048` | 7.9 tok/s | 5.8 tok/s | about 310 tok/s (49 s) | split over both cards |

One card needs the DRAM offload and completed the 15k-token prompt in this test. The V100 has no `causal_conv1d` kernel and no flash-attn, so it uses plain PyTorch paths: the slow, small-memory case, not the speed case.

---

## Author

**Chi Ngai Sam** (Dr. Addie Sam on GitHub), independent researcher.

- Code: [github.com/addiesam/qvb8](https://github.com/addiesam/qvb8) · Models: [huggingface.co/addiesam](https://huggingface.co/addiesam)
- Support the work: [github.com/sponsors/addiesam](https://github.com/sponsors/addiesam)
- License: Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE))
- Issues, bug reports and "it did not reproduce on my GPU" reports: GitHub issues. Tested GPUs are the DGX Spark (GB10) and V100 only; reports from other cards are welcome.

### Papers

1. *Certified L∞-Bounded Compression for AI Models, I: The Contract and the Virtual-Band Index.* Zenodo, 2026-10-30. [doi:10.5281/zenodo.23815007](https://doi.org/10.5281/zenodo.23815007)
2. *Certified L∞-Bounded Compression for AI Models, II: Compressing a Language Model's Weights, Key–Value Cache and Activations.* Zenodo, 2026-10-30. [doi:10.5281/zenodo.23816044](https://doi.org/10.5281/zenodo.23816044)
3. *Certified L∞-Bounded Compression for AI Models, III: Compressing the Model and the Trajectory: a Tau-Collocation ODE Solver for Compressed Diffusion and Video Models.* Zenodo, 2026-10-30. [doi:10.5281/zenodo.23816181](https://doi.org/10.5281/zenodo.23816181)
4. *Applying Certified L∞-Bounded Compression to LLM Serving: Portable Saved State, Shared Pools and Mixed-Generation GPUs.* Zenodo, 2026-11-16. [doi:10.5281/zenodo.24267208](https://doi.org/10.5281/zenodo.24267208)
5. *Certified L∞-Bounded Compression of Large Language Model Weights via On-the-Fly Piecewise Chebyshev Approximation.* [ResearchGate](https://www.researchgate.net/publication/408496032_Certified_L_infty-Bounded_Compression_of_Large_Language_Model_Weights_via_On-the-Fly_Piecewise_Chebyshev_Approximation)

Papers 1 and 2 describe the compression method (the contract, the Virtual-Band Index, weights, KV cache and activations); paper 4 describes the serving system in this file (saved state that moves between machines, shared pools, mixed-generation GPUs); paper 3 extends the method to diffusion and video models.

### Acknowledgements

This work was conducted entirely by the author personally, as a hobby in non-working hours, on the author's own devices. The ideas and mathematical formulation are the author's own, developed through discussion with AI models. The author also used an ensemble of frontier AI models, including the compressed model discussed in this series, to assist in writing the papers and implementing the code. All work and empirical measurements were carried out on the author's own machine.
