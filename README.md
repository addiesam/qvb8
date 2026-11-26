# QVB8 Inference Engine

**Run models on a new foundation: big models, small operating VRAM, certified.**

> 💖 **Free and open. [Sponsor Dr. Addie Sam on GitHub](https://github.com/sponsors/addiesam) to unlock the next models and tests.**

QVB8 uses certified L∞-bounded compression to run massive AI models on limited hardware at near-bf16 quality. Every weight, cache and state stays within a user-defined worst-case error bound (τ), and models run directly from their packed size, with no VRAM expansion.

| **4.0x smaller: 54 GB → 13.6 GB** | **2.5x smaller: 54 GB → 21.6 GB, +0.08% perplexity** | **Real work, no round limit** |
|---|---|---|
| Qwen3.8-27B (QVB8R, τ = 2e-3) on one 16 GB card (perplexity +4.3%) | Qwen3.8-27B (QVB8, τ = 1e-3), WikiText-2 | Full system builds via an OpenAI-compatible API + Cline (600+ rounds and going) |

## A new foundation: the Virtual Band Index

Normal compression is like squeezing a sponge and hoping the water you need stays in. Nobody knows how much was lost in each place.

The **Virtual Band Index** works differently. Think of a ruler. Instead of saving every number exactly, we save which mark on the ruler it is closest to. The marks are so close together that the number can be off by only a tiny, fixed amount. We call that amount **τ** (tau).

Then we **check every single number** after compressing, and write down the biggest mistake we found. **Math guarantees it: no number is ever wrong by more than τ.** That is what **certified** means: the error is controlled by you, and proven, not guessed.

You pick τ. A small τ keeps the model almost the same as the original. A bigger τ makes the file smaller. The same idea also compresses the memory the model uses while it is talking (the KV cache), so the model needs less GPU memory to run, not only less disk. See the [papers](#papers).

## Key Capabilities

- **A full serving engine, not just a format:**
  - **One speculative-decoding layer for every model:** it measures the drafter's acceptance and the card's step cost, so a badly predicted text costs no extra time.
  - **CUDA-graph decoding** on fixed request slots.
  - **Tensor / data / pipeline parallel** across mixed GPUs, on one machine or many.
  - **An OpenAI-compatible API**, with a router in front of any number of servers.
- **Zero-expansion operating VRAM (2.5x to 4x smaller):**
  - A model occupies only its packed size in GPU memory during inference: no full-size expansion, no OOM.
  - Qwen3.8-27B 54 GB → 13.6 GB (4.0x); Ripple-65B-Preview 130 GB → 38.1 GB (3.4x); MiniMax-H3 video transformer 65.7 GB → 26.6 GB (2.5x).
- **Certified worst-case accuracy:**
  - Standard quantization optimizes the average error; here every weight is guaranteed to stay within your chosen error bound (τ).
  - You pick τ: a small τ stays close to the original, a larger τ gives a smaller file.
  - **No codebook to store:** each row is banded against a *virtual* codebook, so a weight is restored by one fused multiply-add (`μ + i·τₛ`), with its few parameters stored once per row.
  - **The proof is the number the GPU writes:** the deployed kernel's `fmaf` was checked bit for bit against a true single-rounding fused multiply-add (Decimal arithmetic) over 2×10⁵ trials, so the certificate on paper is the value in memory, not an approximation of it.
  - **A proven floor:** any certified, randomly accessible representation needs at least ⌈log₂ N_τ⌉ bits per element (N_τ = the τ-covering number of the weights); the codec is built around this bound.
  - **4-bit codes with a guarantee:** on real Qwen3.8-27B embedding and down-projection weights, a 4-bit linear code at τ = 5e-3 escapes at most 0.008% of elements and holds 4.50–4.55 bits per element (3.5x smaller than bf16), with a machine-checked worst-case bound that a plain 4-bit code does not carry.
  - **Measured on a small model:** at τ = 1e-3 on Qwen3.5-0.8B, GSM8K accuracy is statistically indistinguishable from dense bf16 and WikiText-2 / C4 perplexity move by about one percent. See the [papers](#papers).
- **One card or mixed clusters:**
  - **One small card:** fit a large model on a single 16 GB GPU (Qwen3.8-27B or MiniMax-H3 on one V100).
  - **Many cards, many machines:** run in parallel or split across different GPU types (DGX Spark, V100, H100, H200, RTX).
  - **Same result everywhere:** work prefilled on one card (GB10) is restored on another (V100) with the same answer.
- **Full-pipeline compression:**
  - One method packs every component: transformers, text encoders, VAEs, UNets, DFlash drafters, embedding and reranker models, voice models, LoRAs.
  - So the whole stack fits, not only the language model.
- **A KV cache that is certified, small, on-the-fly and shareable:**
  - **Certified:** every cached key and value stays within the error bound (τ), the same guarantee as the weights, and so do the recurrent linear-attention state (Gated-DeltaNet) and the attention. No token is dropped or approximated by eviction.
  - **Small:** about 26.5 KB per token on Qwen3.8-27B (KV τ = 0.2) against 64 KB in 16-bit. 150k tokens take 4.1 GB instead of 9.8 GB, at +0.006% perplexity.
  - **On the fly:** each new token is compressed as it is generated, and the attention reads the compressed cache directly, so the cache never has to be expanded to 16-bit first.
  - **Can be split over machines:** older KV and saved prompt prefixes move to system RAM or to a DRAM PC, and a prefix computed on one machine (a GB10) is restored on another (a V100) with the same result.
- **Built for production workflows:**
  - **Autonomous software engineering:** an OpenAI-compatible API driving VS Code + Cline for 600+ rounds.
  - **End-to-end media pipelines:** script, images, video and voice on limited hardware, because every model in the chain is compressed.
  - **Long history:** older history moves to system RAM, so a long session keeps going.

## What's new in v1.3.0

- **Two more model families:** [SenseNova-Loop-9B](scripts/sensenova_server.sh) (a looped hybrid: layers run twice with two mixed residual streams) with its DFlash2 drafter, and [Inkling-2-Mini](scripts/inkling_server.sh) (a looped mixture-of-experts model, experts in their published NVFP4 / MXFP8 form, bit-exact) with the DSpark drafter bundled in its checkpoint (`--drafter dspark`).
- **One speculative-decoding layer for every model:** each step the server decides how many drafts to verify from the drafter's measured acceptance and the card's measured step cost, so text the drafter predicts badly costs no extra time. The same layer serves one card, several cards in one process, one process per card, and image prompts.
- **CUDA-graph speculative decoding** on fixed request slots, sized at start from a measured prefill so that long prompts still fit.
- **Faster looped models:** fused Triton kernels for the small per-token operations (routing, norms, short convolutions).
- **Tensor parallel for Qwen models:** every layer split over the cards of one machine (`scripts/qwen_tp_server.sh`), with temperature / top-k / top-p sampling in every TP mode (single request with a drafter, `USERS=N` conversations together, and pipeline slots). On two V100 16 GB it answered the 27B in 7 s against 16 s on one card.
- **Prebuilt CUDA kernels for V100 (sm_70) and DGX Spark (sm_121a)** are bundled, so neither needs a CUDA toolkit to run.

Start them like the other models (port 8080):

SenseNova-Loop-9B:

```bash
MODEL=./models/SenseNova-Loop-9B_qvb8_5e-4 DRAFT=./models/SenseNova-Loop-9B-DFlash2_qvb8_5e-4 scripts/sensenova_server.sh start
```

Inkling-2-Mini:

```bash
MODEL=./models/Inkling-2-Mini_qvb8_1e-3 scripts/inkling_server.sh start
```

Qwen3.8-27B by tensor parallel (two cards):

```bash
MODEL=./models/Qwen3.8-27B_qvb8_1e-3_fp16 DRAFT=./models/Qwen3.8-27B-DFlash2_qvb8_2e-3_fp16 scripts/qwen_tp_server.sh start
```

## Compression and Fidelity

### Language models

| Model | τ | Original (bf16) | QVB8 packed (ratio) | QVB8R packed (ratio) | Perplexity vs bf16 |
|---|---|---|---|---|---|
| [Qwen3.8-27B](https://huggingface.co/collections/addiesam/qwem38-27b) | 5e-4 | 54 GB | 25.2 GB (2.1x) | 20.8 GB (2.6x) | +0.18% (7.0641 vs 7.0517) |
| [Qwen3.8-27B](https://huggingface.co/collections/addiesam/qwem38-27b) | 1e-3 | 54 GB | 21.6 GB (2.5x) | 17.2 GB (3.1x) | +0.08% (7.0570 vs 7.0517) |
| [Qwen3.8-27B](https://huggingface.co/collections/addiesam/qwem38-27b) | 2e-3 | 54 GB | 18.8 GB (2.9x) | 13.6 GB (4.0x) | +4.3% (7.3549 vs 7.0517) |
| [Qwen3.8-27B DFlash2](https://huggingface.co/collections/addiesam/qwem38-27b) | 2e-3 | 3.9 GB | 2.1 GB (1.8x) | [sponsor to calibrate](https://github.com/sponsors/addiesam) | — |
| [Qwen3.8-27B DFlash2](https://huggingface.co/collections/addiesam/qwem38-27b) | 5e-3 | 3.9 GB | 1.7 GB (2.3x) | 1.5 GB (2.6x) | — |
| [Qwen3.6-35B-A3B](https://huggingface.co/collections/addiesam/qwen36-35b-a3b) | 1e-4 | 70 GB | 42.9 GB (1.6x) | [sponsor to calibrate](https://github.com/sponsors/addiesam) | 0.00% (10.690 vs 10.690, C4) |
| [Qwen3.6-35B-A3B](https://huggingface.co/collections/addiesam/qwen36-35b-a3b) | 5e-4 | 70 GB | 31.5 GB (2.2x) | 29.5 GB (2.4x) | +0.13% (C4) |
| [Qwen3.6-35B-A3B](https://huggingface.co/collections/addiesam/qwen36-35b-a3b) | 1e-3 | 70 GB | 27.0 GB (2.6x) | [sponsor to calibrate](https://github.com/sponsors/addiesam) | +0.70% (C4) |
| [Qwen3.6-35B-A3B DFlash2](https://huggingface.co/collections/addiesam/qwen36-35b-a3b) | 1e-3 | 1.1 GB | 0.6 GB (1.7x) | [sponsor to calibrate](https://github.com/sponsors/addiesam) | — |
| [Ripple-65B-Preview](https://huggingface.co/collections/addiesam/ripple-65b-preview-qvb8) | 1e-3 | 130 GB | 57.3 GB (2.3x) | 46.6 GB (2.8x) | [sponsor to measure](https://github.com/sponsors/addiesam) |
| [Ripple-65B-Preview](https://huggingface.co/collections/addiesam/ripple-65b-preview-qvb8) | 2e-3 | 130 GB | 49.1 GB (2.6x) | 38.1 GB (3.4x) | [sponsor to measure](https://github.com/sponsors/addiesam) |
| [Ripple-65B-Preview](https://huggingface.co/collections/addiesam/ripple-65b-preview-qvb8) | 2.5e-3 | 130 GB | 48.1 GB (2.7x) | 35.5 GB (3.7x) | [sponsor to measure](https://github.com/sponsors/addiesam) |
| [Ripple-65B-Preview DFlash2](https://huggingface.co/collections/addiesam/ripple-65b-preview-qvb8) | 5e-3 | 8.1 GB | 4.1 GB (2.0x) | 3.4 GB (2.4x) | — |
| [SenseNova-Loop-9B](https://huggingface.co/addiesam/SenseNova-Loop-9B-QVB8-5e-4) | 5e-4 | 19 GB | 8.8 GB (2.2x) | [sponsor to calibrate](https://github.com/sponsors/addiesam) | [sponsor to measure](https://github.com/sponsors/addiesam) |
| [SenseNova-Loop-9B](https://huggingface.co/addiesam/SenseNova-Loop-9B-QVB8-1e-3) | 1e-3 | 19 GB | 7.7 GB (2.5x) | [sponsor to calibrate](https://github.com/sponsors/addiesam) | [sponsor to measure](https://github.com/sponsors/addiesam) |
| [SenseNova-Loop-9B DFlash2](https://huggingface.co/addiesam/SenseNova-Loop-9B-DFlash2-QVB8-5e-4) | 5e-4 | 1.9 GB | 0.9 GB (2.1x) | [sponsor to calibrate](https://github.com/sponsors/addiesam) | — |
| [Inkling-2-Mini](https://huggingface.co/addiesam/Inkling-2-Mini-QVB8-5e-4) | 5e-4 | 19 GB | 18 GB (1.1x) | [sponsor to calibrate](https://github.com/sponsors/addiesam) | [sponsor to measure](https://github.com/sponsors/addiesam) |
| [Inkling-2-Mini](https://huggingface.co/addiesam/Inkling-2-Mini-QVB8-1e-3) | 1e-3 | 19 GB | 18 GB (1.1x) | [sponsor to calibrate](https://github.com/sponsors/addiesam) | [sponsor to measure](https://github.com/sponsors/addiesam) |

### Image, video and audio models (unreleased: published once the [donation](https://github.com/sponsors/addiesam) target is reached)

| Model / component | τ | Original (bf16) | QVB8 packed (ratio) | QVB8R packed (ratio) |
|---|---|---|---|---|
| MiniMax-H3 (full package) | 1e-2 / 1e-3 | 137.6 GB | 57.8 GB (2.4x) | 45.9 GB (3.0x) |
| - Video transformer | 1e-2 | 65.7 GB | 26.6 GB (2.5x) | 21.1 GB (3.1x) |
| - 33B text encoder | 1e-3 | 66.7 GB | 29.1 GB (2.3x) | 23.0 GB (2.9x) |
| - VAE | 1e-3 | 5.2 GB | 2.1 GB (2.5x) | 1.8 GB (2.9x) |
| MiniMax-H3 low_ram (full package) | 1e-2 / 3e-3 / 1e-3 | 137.6 GB | 50.5 GB (2.7x) | 39.2 GB (3.5x) |
| - Video transformer | 1e-2 | 65.7 GB | 26.6 GB (2.5x) | 21.1 GB (3.1x) |
| - 33B text encoder | 3e-3 | 66.7 GB | 21.8 GB (3.1x) | 16.3 GB (4.1x) |
| - VAE | 1e-3 | 5.2 GB | 2.1 GB (2.5x) | 1.8 GB (2.9x) |
| Qwen-Image-Edit | 1e-3 | 58.0 GB | 29.9 GB (1.9x) | 26.3 GB (2.2x) |
| Wan2.2-TI2V-5B | 1e-3 | 24.2 GB | 12.7 GB (1.9x) | 11.5 GB (2.1x) |
| Sonic UNet | 1e-3 | 6.3 GB | 1.4 GB (4.5x) | 1.4 GB (4.5x) |

MiniMax-H3 package = video transformer + text encoder + VAE (the 0.6 GB audio decoder is kept dense and not counted). Sizes are the packs on disk.

## Run AI Models using QVB8 Inference Engine

### General setup

Linux with an NVIDIA GPU and its driver is all you need.

**1. Get the code and install** (about 10 minutes, creates `.venv`, no CUDA toolkit needed):

```bash
git clone https://github.com/addiesam/qvb8.git && cd qvb8
./install.sh
```

**2. Download a model from Hugging Face** into `./models` (download and save all model files to a folder; the download resumes if it stops):

```bash
hf download addiesam/<model card> --local-dir ./models/<model card>
```

### Scenario 1: Quick Try Qwen3.5-9B using the QVB8 Inference Engine

**Use when:** you want to try the engine in about 10 minutes on any NVIDIA card with 16 GB or more.

**Download the models:** (download and save all model files to a folder)

| Model type | Model | Size | Hugging Face (download) |
|---|---|---|---|
| LLM | Qwen3.5-9B (τ = 1e-3, QVB8) | 7.6 GB | [addiesam/Qwen3.5-9B_qvb8_1e-3_fp16](https://huggingface.co/addiesam/Qwen3.5-9B_qvb8_1e-3_fp16) |
| DFlash | Qwen3.5-9B DFlash (τ = 1e-3, QVB8) | 1.4 GB | [addiesam/Qwen3.5-9B-DFlash_qvb8_1e-3](https://huggingface.co/addiesam/Qwen3.5-9B-DFlash_qvb8_1e-3) |

```bash
hf download addiesam/Qwen3.5-9B_qvb8_1e-3_fp16 --local-dir ./models/Qwen3.5-9B_qvb8_1e-3_fp16
hf download addiesam/Qwen3.5-9B-DFlash_qvb8_1e-3 --local-dir ./models/Qwen3.5-9B-DFlash_qvb8_1e-3
```

Or download both with one command:

```bash
./qvb8 get qwen-9b
```

**Run:**

```bash
# starts the QVB8 Inference server (OpenAI-compatible mode) on port 8080 (first start: a few minutes)
./qvb8 start qwen-9b
```

**Chat in the terminal:**

```bash
./qvb8 chat
```

**Stop the server:**

```bash
./qvb8 stop
```

---

### Scenario 2: runs Qwen3.8-27B on one 16 GB card (like a V100)

**Use when:** you have one 16 GB card (like a V100) and want the 27B model.

**Download the models:** (download and save all model files to a folder)

| Model type | Model | Size | Hugging Face (download) |
|---|---|---|---|
| LLM | Qwen3.8-27B (τ = 2e-3, QVB8R) | 13.6 GB | [addiesam/Qwen3.8-27B_qvb8r_2e-3_fp16](https://huggingface.co/addiesam/Qwen3.8-27B_qvb8r_2e-3_fp16) |
| DFlash | Qwen3.8-27B DFlash2 (τ = 5e-3, QVB8R) | 1.5 GB | [addiesam/Qwen3.8-27B-DFlash2_qvb8r_5e-3_fp16](https://huggingface.co/addiesam/Qwen3.8-27B-DFlash2_qvb8r_5e-3_fp16) |

```bash
hf download addiesam/Qwen3.8-27B_qvb8r_2e-3_fp16 --local-dir ./models/Qwen3.8-27B_qvb8r_2e-3_fp16
hf download addiesam/Qwen3.8-27B-DFlash2_qvb8r_5e-3_fp16 --local-dir ./models/Qwen3.8-27B-DFlash2_qvb8r_5e-3_fp16
```

Or download both with one command:

```bash
./qvb8 get qwen-27b
```

**Run:**

```bash
# starts the QVB8 Inference server (OpenAI-compatible mode) on port 8080 (first start: a few minutes); the 16 GB card options are set for you
./qvb8 start qwen-27b

# the same start by hand
MODEL=./models/Qwen3.8-27B_qvb8r_2e-3_fp16 DRAFT=./models/Qwen3.8-27B-DFlash2_qvb8r_5e-3_fp16 DRAFT_K=7 \
QVB8_MLP_GRAPH=1 QV_HOST_EMBED=1 QV_DRAFT_CTX=2048 ATTN=sdpa MAX_LIVE=1 scripts/qwen_server.sh start \
  --prefill-chunk 512 --kv-window 4096 --kv-cold-block 8192 --prefix-store host --vision-tower host
```

**Chat in the terminal:**

```bash
./qvb8 chat
```

**Stop the server:**

```bash
./qvb8 stop
```

**Remark: long context needs the DRAM KV cache.** The 13.6 GB model leaves only a little VRAM on a 16 GB card, so the KV cache must live in system RAM (DRAM). `--kv-window 4096` keeps only the newest 4096 tokens in VRAM, `--kv-cold-block 8192` moves the older history to DRAM, and `--prefix-store host` keeps the saved prompt prefixes in DRAM. This is how a long context works on 16 GB: 150k tokens of KV take about 4 GB of DRAM (26.5 KB per token), so keep several GB of system RAM free.

**Use it from your tools:** the server speaks the OpenAI API at `http://localhost:8080/v1`. In **VS Code + Cline**
choose "OpenAI Compatible" and set that base URL.

---

### Scenario 3: a pair of 16 GB cards (two V100) runs Qwen3.8-27B at a tighter τ

**Use when:** you have two 16 GB cards in one PC and want a tighter τ (closer to bf16).

**Machine:** one PC with two 16 GB cards (two V100), one server process for both.

A larger pack (QVB8 τ = 1e-3, 21.6 GB, perplexity +0.08%) is too big for one 16 GB card, so the layers are split over both cards in one process (about 11 GB per card, measured). Download `addiesam/Qwen3.8-27B_qvb8_1e-3_fp16` and `addiesam/Qwen3.8-27B-DFlash2_qvb8_2e-3_fp16` into `./models` as above, then:

```bash
MODEL=./models/Qwen3.8-27B_qvb8_1e-3_fp16 DRAFT=./models/Qwen3.8-27B-DFlash2_qvb8_2e-3_fp16 DRAFT_K=7 \
QVB8_MLP_GRAPH=1 QV_HOST_EMBED=1 ATTN=sdpa scripts/qwen_server.sh start --pipeline auto
```

`--pipeline auto` uses every visible card and places the layers by weight size and free memory; cards of different sizes work. Choose the cards yourself with `--pipeline cuda:0,cuda:1`.

---

### Scenario 4: SSD + DRAM on one machine: a long context or a big model beyond the VRAM

**Use when:** the context or the model is bigger than the VRAM, and the machine has RAM and an SSD.

**Machine:** one GPU machine with enough system RAM and an SSD.

- **LLM, DRAM:** older KV, saved prefixes and the vision tower live in system RAM (`--kv-window`, `--prefix-store host`, `--vision-tower host`, as in Scenario 2).
- **Image and video models, SSD + DRAM:** the transformer blocks that do not fit the card are streamed from the SSD, with an optional DRAM buffer of 8 to 12 GB. Example: MiniMax-H3 (low_ram pack) on one 16 GB V100 made a 1280x704 video in 46.5 minutes.

---

### Scenario 5: a DRAM PC as the pool for the GPU machines

**Use when:** several GPU machines need one big RAM pool for older KV and saved prefixes.

A cheap PC with a lot of RAM and no GPU holds what the GPU machines cannot: the older KV of long requests and a copy of every saved prefix. The GPU memory then goes to live work, a restart of a GPU server loses nothing, and the PC needs only plain Python (no GPU, no storage service).

| Machine | Hardware | Runs |
|---|---|---|
| DRAM PC | any PC with a lot of RAM, no GPU | `prefix_store` (copies of saved prefixes) and `kv_host` (older KV blocks) |
| GPU machine A | for example a V100 (16 GB) or a DGX Spark | the model server, with `--prefix-backup` and `--kv-remote` pointing at the PC |
| GPU machine B (optional) | any other generation | the same, so a prefix saved on A can be restored on B |

```bash
# on the DRAM PC (standard library only)
python -m ai_engine.servers.prefix_store --mirror http://hostA:8080 --port 8082
python -m ai_engine.servers.kv_host --port 8083 --budget-gb 8
# on each GPU machine
scripts/qwen_server.sh start --prefix-backup http://pc:8082 --kv-remote http://pc:8083 --kv-window 4096
```

---

### Scenario 6: across machines, no matter new or old cards (architecture)

**Use when:** you have machines with different GPU generations and want one service.

Machines of different generations serve as one service: for example a DGX Spark (Blackwell, GB10) and a box of V100 cards (Volta).

| Machine | Hardware | Runs |
|---|---|---|
| Machine 1 | DGX Spark (GB10) | a model server (port 8080) and the router (port 8000) |
| Machine 2 | box with V100 cards | one model server per card (ports 8080 and 8081), the same pack as machine 1 |
| DRAM PC (optional) | RAM, no GPU | the prefix store, so a prefix computed on one machine can be pulled by another (Scenario 5) |

Usage cases:

- **A team shares one coding service.** Every machine runs the same pack and one router URL sits in front. Each request goes to the machine that already holds its saved prefix, or the prefix is copied there when copying is faster than computing. A 94k-token prompt took 269 s on a V100 pair the first time and 66 s from the saved prefix; a repeated prompt sent through the router to the V100 that held it took 4.3 s.
- **Prefill on the fast machine, continue on the old one.** A prompt prefilled on the GB10 is restored on a V100 with the same answer and no recompute.
- **A model too big for one machine.** Run a stage worker for a range of layers on another machine and add it to the pipeline. Set the same `QV_POOL_TOKEN` on both. Tested: two V100 cards and a GB10 over Wi-Fi.

```bash
# on the other machine: runs layers 0 to 12
python -m ai_engine.cli.stage_worker $MODEL --stage-port 8081 --stage-layers 0:12
# on this machine: the rest of the layers
scripts/qwen_server.sh start --pipeline host:8081,local --pipeline-split 12
```

```bash
# one router URL for every client, over replicas on different machines
python -m ai_engine.servers.router --port 8000 \
    --backend gb10=http://127.0.0.1:8080 --backend v100c1=http://v100host:8080 --backend v100c0=http://v100host:8081
```

---

### Scenario 7: data center, new and old cards co-work (H100, H200, B200, B300, V100 and older)

**Use when:** you run a data center with new servers and older boxes together.

Current server cards and older cards work in the same deployment, because the saved state and the packs are defined by their numbers and not by a GPU.

| Machine | Hardware | Runs |
|---|---|---|
| New GPU servers | H100, H200, B200, B300 | the large models (one model over several cards, `--pipeline auto`) and the long-prompt replicas |
| Older GPU boxes | V100 and other older cards | chat and agent replicas of the same pack, one per card |
| DRAM pool machines | a lot of RAM, no GPU | the prefix store and the older KV (Scenario 5) |
| Router machine | any CPU machine | the router, one URL for every client |

Usage cases:

- **A mixed fleet behind one router.** New cards (H100, H200, B200, B300) take the big models and the long prompts; older cards (V100 and the like) take the chat and agent replicas. The router chooses a replica by queue, saved-prefix hit and link speed.
- **One huge model over many cards.** The layers are placed over the cards of one machine automatically (`--pipeline auto`), and over several machines with stage workers. On one 80 GB card, a 65B model runs as it is: Ripple-65B-Preview (QVB8, τ = 1e-3, 57.3 GB).

```bash
hf download addiesam/Ripple-65B-Preview-QVB8-1e-3 --local-dir ./models/Ripple-65B-Preview-QVB8-1e-3
hf download addiesam/Ripple-65B-Preview-DFlash2-QVB8-5e-3 --local-dir ./models/Ripple-65B-Preview-DFlash2-QVB8-5e-3
```

Then start the server:

```bash
MODEL=./models/Ripple-65B-Preview-QVB8-1e-3 DRAFT=./models/Ripple-65B-Preview-DFlash2-QVB8-5e-3 scripts/ripple_server.sh
```

- **A super DRAM pool.** Machines with a lot of RAM and no GPU keep the saved prefixes and the older KV, so the GPU memory goes to live work and fewer GPUs are needed (Scenario 4).
- **Many clusters in parallel.** Each cluster is one model over several GPUs; every cluster offloads to the shared DRAM pool.

**Status:** tested on a DGX Spark (GB10) and V100 cards. H100, H200, B200 and B300 servers, and three or more cards, have not been tested: the layouts above are simulations. Prebuilt kernels are included for the GB10 (sm_121a) and the V100 (sm_70); on any other GPU the kernels compile on first use and need the CUDA toolkit. To get these tested, see [Support this project](#sponsor-this-project).

---

### Scenario 8: a team shares long documents and agent rounds

**Use when:** a team or an agent asks about the same long text again and again.

An agent (Cline, OpenCode) sends the whole history again in every round, and a team often asks about the same long document. The server saves the computed state of the beginning of the history, so the next round, or the next person, prefills only what is new.

| Machine | Runs | Why |
|---|---|---|
| One GPU machine, several servers | `--prefix-store shared` | the servers of one machine share one RAM pool: a document prefilled on card 0 is restored on card 1, stored once |
| Several GPU machines + the DRAM PC | the router, `--prefix-backup` on every server | a prefix saved on one machine is pulled by another when copying is faster than computing |

---

### Scenario 9: many users at once (throughput): replicas of a model split by layers

**Use when:** many users send messages at the same time, or one model is too big for one machine and you also need more requests per second.

**One message does not get faster by adding machines**: its tokens pass through the layers in order, so splitting by layers adds memory, not speed. What makes one message faster is the DFlash drafter, a saved prefix, and fewer hops (fewer machines). **More machines serve more messages at the same time.** Build it in two steps:

1. **Split by layers inside each machine.** One server process per machine puts the layers on its cards in order (first layers on card 0, the next on card 1, and so on). You do not need one server per layer range.
2. **Copy the machine, put one router in front.** Machines 2 and 3 run the same thing, and the router sends each new message to a free machine (or to the one that already holds its saved prefix).

| Machine | Hardware | Runs |
|---|---|---|
| Machine 1 | 3 cards | one server, `--pipeline cuda:0,cuda:1,cuda:2`: layers split over its 3 cards |
| Machine 2 | 3 cards, same | the same server and the same pack |
| Machine 3 | 3 cards, same | the same server and the same pack |
| Router machine | any CPU machine | the router, one URL for every client |

```bash
# on machine 1, 2 and 3 (the same command)
MODEL=./models/<pack> DRAFT=./models/<drafter pack> DRAFT_K=7 scripts/qwen_server.sh start --pipeline auto
# on the router machine
python -m ai_engine.servers.router --port 8000 \
    --backend m1=http://machine1:8080 --backend m2=http://machine2:8080 --backend m3=http://machine3:8080
```

Three machines serve three times as many messages at once. Inside each machine, the continuous-batching scheduler also serves several users together (`--max-live`). Not measured yet: the scaling beyond three replicas, and several requests overlapping inside one layer-split pipeline.

## Sponsor this project

<div align="center">

# Every sponsor helps me publish more models, for free.

# 👉 [**SPONSOR DR. ADDIE SAM ON GITHUB**](https://github.com/sponsors/addiesam) 👈

[![Sponsor](https://img.shields.io/badge/SPONSOR-GitHub%20Sponsors-ea4aaa?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/addiesam)

</div>

QVB8 is built and tested by one person on a few old and new GPUs, and every pack, kernel and paper is free. Your sponsorship decides what comes next:

| Your support | What it unlocks |
|---|---|
| **Sponsor on GitHub** | the [image, video and audio model packs](#image-video-and-audio-models-unreleased-published-once-the-donation-target-is-reached) are published once the donation target is reached; more language-model packs and faster releases |
| **A GPU card, or GPU time** | testing on H100 / H200 / H300 servers and on three or more cards (today only a DGX Spark and V100 cards are tested), and the TP + DP numbers that are still "not measured" |
| **Server access or rented GPU hours** | long runs: perplexity of the packs still marked "sponsor to measure", and the data center layouts that are simulations today |
| **Companies** | custom model packs (other models, other GPUs) and support |

**Contact:** open an issue or write to the author through [github.com/addiesam](https://github.com/addiesam).

If QVB8 saved you a GPU, please [sponsor](https://github.com/sponsors/addiesam). Even a small amount keeps it going.

## Every way to start the API server (one card, TP, DP, PP, TP + DP, DRAM, disk, many machines)

**Use when:** you want to pick how the server uses your cards. Every row starts the same OpenAI-compatible API (`http://host:port/v1`); only the start command differs. A model that does not fit is refused at start with the numbers, not later.

| Mode | What it does | Best for | Start (Qwen3.8-27B; add `MODEL=` and `DRAFT=` as in Scenario 2 / 3) |
|---|---|---|---|
| **Single card** | one card holds the model | a model that fits one card | `scripts/qwen_server.sh start` |
| **PP**: layers split | layers 0..n on card 0, the rest on card 1, one process | a model too big for one card | `scripts/qwen_server.sh start --pipeline cuda:0,cuda:1` |
| **DP**: one server per card + router | each card holds the whole model; the router sends each request to a free card | many users at once | two single-card servers on ports 8081 / 8082, then `python -m ai_engine.servers.router --port 8080 --backend a=http://localhost:8081 --backend b=http://localhost:8082` |
| **TP**: tensor parallel | every layer split over the cards (heads and experts), in one process | one request as fast as possible (measured 7 s vs 16 s on one card for Qwen3.8-27B on 2 x V100) | `scripts/qwen_tp_server.sh start` (Qwen family, 2 cards); `QV_TP_DEVICES=cuda:0,cuda:1 scripts/inkling_server.sh start` (Inkling-2-Mini) |
| **TP + DP** | each router backend is itself a TP group (4 cards = 2 groups of 2) | fast single requests and many users | start one TP server per card pair (`CUDA_VISIBLE_DEVICES=0,1` and `2,3`, ports 8081 / 8082), then the router as in DP. *Not yet run on four cards.* |
| **DRAM** | older KV, saved prefixes and the vision tower in system RAM | long context on a small card | `--kv-window 4096 --prefix-store host --vision-tower host` (Scenario 2) |
| **DRAM + disk** | experts or blocks that do not fit the card stay in host RAM or are streamed from the SSD | a model bigger than VRAM + RAM | `QV_EXPERTS=host scripts/inkling_server.sh start` (experts in host RAM) |
| **DRAM PC pool** | a RAM-only PC keeps saved prefixes and older KV for all GPU machines | many machines, long shared documents | `--prefix-backup http://pc:8082 --kv-remote http://pc:8083` (Scenario 5) |
| **Many machines, different cards (replicas)** | every machine runs the whole model on its own cards (a GB10, V100 boxes, RTX ...); one router URL in front. A prefix prefilled on a fast machine is restored on an old card with the same answer | a team or a data center mixing new and old GPUs | on each machine `scripts/qwen_server.sh start` (one server per card, own port), then `python -m ai_engine.servers.router --port 8000 --backend gb10=http://gb10:8080 --backend v100a=http://v100host:8080 --backend v100b=http://v100host:8081` (Scenarios 6, 7, 9) |
| **Many machines, one big model (layers split)** | machine 1 runs layers 0..n, machine 2 the rest, whatever the cards are | a model too big for one machine | on machine 2 `python -m ai_engine.cli.stage_worker $MODEL --stage-port 8081 --stage-layers 0:12`, on machine 1 `scripts/qwen_server.sh start --pipeline host:8081,local --pipeline-split 12` (Scenario 6; set the same `QV_POOL_TOKEN` on both) |

Combine them: for example, DRAM + PP on two cards, or a router over machines that each run PP. On two V100 16 GB, [every mode was measured](#two-16-gb-v100-every-parallel-mode-measured). Qwen-family TP is `scripts/qwen_tp_server.sh` (single-message speed path by default, `USERS=N` to serve N conversations together; greedy or sampled per request).

### Two 16 GB V100: every parallel mode, measured

One machine, 2 × V100 16 GB (no P2P between the cards), 31 GB host RAM. Each cell is one real request through the
local API, including the server's own per-request work. Modes: **single** = one card; **dual (PP)** = the layers split
over both cards; **TP** = tensor parallel over both cards; **pair (DP)** = one server per card behind one router URL.

| Model | Request | single | dual (PP) | TP | pair (DP) |
|---|---|---|---|---|---|
| Z-Image-Turbo | 832×480 image | ✅ 8 s | ✅ 8 s | ✅ 10 s | ✅ 6–8 s |
| Qwen3.8-27B | ~110-token reply | ✅ 16 s | ✅ 13 s | ✅ 7 s | ✅ 14–45 s |
| Qwen-Image-Edit-2511 | 832×480, 3 reference images | ✅ 162 s (3e-3 text encoder, 8 GB RAM buffer) | ✅ 114 s | ✅ 287 s | ✅ 510 s; two at once: 300 s and 714 s |
| MiniMax-H3 | 832×480 video + audio, 124 frames, first-frame | ✅ 400 s | ✅ 841 s | ✅ 723 s | ✅ 851 s; two at once: 300 s and 660 s |
| Sonic (lip-sync) | 4 s clip | ✅ 131 s | — no card split for this server | — | ✅ 130 s (684 s on the throttled card) |
| Qwen3-ASR / ForcedAligner / TTS | 16 s Cantonese clip / same / one sentence | ✅ 3 s / 1 s / 20 s | — small models, one card is enough | — | ✅ 3 s / 1 s / 9–19 s |
| Qwen3.5-9B | 120-token reply (first request) | ✅ 19 s (warm: 14–17 tok/s) | ✅ 7 s | ✅ 3 s | ✅ 6 s |
| SenseNova-Loop-9B | ~85-token reply | ✅ 29 s | ✅ 39 s | ✅ 16 s | ✅ 36 s |
| Inkling-2-Mini | 120-token reply | ✅ 109 s | ✅ 101 s | ✅ 16 s | — two copies need more than 31 GB of host RAM |
| Qwen3.6-35B-A3B | — | — the 30 GB pack needs more than one card | refused at start: needs 29.7 GiB, the cards have 2 × 13 | refused at start: 14.9 GiB per card, 13 free | — as single |

On this board one card ran hot and throttled, which explains the spread in the pair (DP) times. Card-to-card copies
go through host memory here, so TP does not beat one card for a model that fits a single card (Z-Image). A model
that cannot fit is refused when the server starts, with the numbers, instead of failing later.

On a DGX Spark (GB10, one GPU, 121 GB unified memory) the same servers were checked with one small request each,
beside other running jobs: Z-Image, Wan2.2-TI2V-5B, Qwen-Image-Edit, Sonic, Qwen3-ASR / ForcedAligner / TTS,
Qwen3.5-9B, SenseNova-Loop-9B, Qwen3.8-27B, Inkling-2-Mini and Qwen3.6-35B-A3B all answered.


Design (architecture, every serving mode with diagrams, each key capability in detail) and reference (install, options, measured speed): [DESIGN.md](DESIGN.md).

## Papers

Five papers by the author describe the method and this system. Papers 1 and 2 describe the certified compression method (every restored weight within error bound τ of the original) and its use on language models; paper 3 extends it to diffusion and video models; paper 4 describes the serving system (saved state that moves between machines); paper 5 describes on-the-fly piecewise Chebyshev approximation.

> 1. Sam, Chi Ngai. *Certified L∞-Bounded Compression for AI Models, I: The Contract and the Virtual-Band Index.*
> Zenodo, 2026. [doi:10.5281/zenodo.23815007](https://doi.org/10.5281/zenodo.23815007)

> 2. Sam, Chi Ngai. *Certified L∞-Bounded Compression for AI Models, II: Compressing a Language Model's Weights, Key–Value Cache and Activations.*
> Zenodo, 2026. [doi:10.5281/zenodo.23816044](https://doi.org/10.5281/zenodo.23816044)

> 3. Sam, Chi Ngai. *Certified L∞-Bounded Compression for AI Models, III: Compressing the Model and the Trajectory: a Tau-Collocation ODE Solver for Compressed Diffusion and Video Models.*
> Zenodo, 2026. [doi:10.5281/zenodo.23816181](https://doi.org/10.5281/zenodo.23816181)

> 4. Sam, Chi Ngai. *Applying Certified L∞-Bounded Compression to LLM Serving: Portable Saved State, Shared Pools and Mixed-Generation GPUs.*
> Zenodo, 2026. [doi:10.5281/zenodo.24267208](https://doi.org/10.5281/zenodo.24267208)

> 5. Sam, Chi Ngai. *Certified L∞-Bounded Compression of Large Language Model Weights via On-the-Fly Piecewise Chebyshev Approximation.*
> [ResearchGate](https://www.researchgate.net/publication/408496032_Certified_L_infty-Bounded_Compression_of_Large_Language_Model_Weights_via_On-the-Fly_Piecewise_Chebyshev_Approximation)

## License

The code is © 2026 Sam, Chi Ngai, licensed under [Apache-2.0](LICENSE) (see [NOTICE](NOTICE)).
Questions and bug reports: please open a [GitHub issue](https://github.com/addiesam/qvb8/issues).

The model weights are not part of this repository and keep their own license.

## Disclaimer — no warranty, no liability

**Acknowledgements.** This work was conducted entirely by the author personally, as a hobby in non-working hours, on the author's own devices. The ideas and mathematical formulation are the author's own, developed through discussion with AI models. The author also used an ensemble of frontier AI models, including the compressed model discussed in this series, to assist in writing the papers and implementing the code. All work and empirical measurements were carried out on the author's own machine.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE, ITS
OUTPUT, OR THE USE OF OR OTHER DEALINGS IN THE SOFTWARE. See sections 7 and 8 of the [LICENSE](LICENSE).

In particular:

- **AI output can be wrong.** Language models can produce incorrect, misleading, offensive or unsafe text and code.
  You are solely responsible for checking any output before you rely on it or act on it.
- **The certified error bound covers the compressed weights only.** It guarantees that the error of every restored
  weight, against the original, is less than the error bound tau. It does not guarantee that the model's answers are correct, safe or identical to
  any other implementation.
- **Not for critical use.** Do not use this software where a failure or a wrong answer could cause death, injury,
  financial loss or damage to property or the environment (for example medical, legal, financial, safety or
  infrastructure decisions) without your own independent validation.
- **Third-party models and software.** Model weights and dependencies (PyTorch, Transformers, flash-attn, and
  others) are provided by third parties under their own licenses. The author makes no
  statement about them and accepts no responsibility for them.
- **Your hardware and data.** You run this software at your own risk, including any effect on your hardware, your
  data, or any service you expose to a network (the server has no authentication: run it on a trusted network).
