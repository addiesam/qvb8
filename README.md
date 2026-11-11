# Certified L∞ Bounded Compression for AI Models - QVB8 Inference Engine

QVB8 stores model weights in a compressed format where **the error of every restored weight is guaranteed to be less than the
error bound τ**. The engine runs the compressed weights directly on the GPU: custom CUDA kernels decode them inside
the matmuls, so large models run on small hardware. It provides an OpenAI-compatible HTTP server and a terminal chat
client, with DFlash2 speculative decoding.

The engine is general CUDA code. **Version 1 provides a build for the NVIDIA DGX Spark (GB10, sm_121, aarch64).**

## Why certified L∞-bounded compression?

Most quantization is judged by an average: a perplexity or benchmark score. An average hides the worst case. A format can
look good on average and still leave a few weights with a large error, and a large error in the wrong weight can flip a
single token. The result is a sudden wrong word or a stray character from another language in the middle of a normal
answer, which an average score may not show. In our own tests with a 4-bit floating-point format we saw this in server logs.

L∞-bounded means the **largest** error is controlled, not the average. For a QVB8 pack:

- every restored weight differs from the original by less than **τ**, and no tensor is allowed to exceed it;
- the check is made after packing, tensor by tensor, and the largest error found is stored in `manifest.json` (`max_err`);
- τ is your choice: a smaller τ is a tighter bound, a larger file and a perplexity closer to the original; a larger τ is
  a smaller file and a higher perplexity. You choose the trade-off knowing the worst case.

What the bound does **not** do: it covers the stored weights only. It does not promise that the model's answers are
identical to the original's, or correct. See the disclaimer at the end of this file.

## What version 1 supports

- **Model type:** large language models (text in, text out), served through a chat API.
- **Model:** one model so far, **Ripple-65B-Preview** (a dense 65B LLM by Ricursive AI), with its DFlash2 draft model for
  speculative decoding.
- **Weight formats:** QVB8 and QVB8R packs (QVB8R is the smaller, entropy-coded variant).
- **Interfaces:** an OpenAI-compatible HTTP server (streaming, tool calls, reasoning output) and a terminal chat client.
- **Hardware:** one GPU, or several in one machine (`--pipeline cuda:0,cuda:1`, one process, the layers split across
  the cards), or several machines (a remote stage worker runs a layer range on another box).
- **KV cache:** the certified compressed KV cache, with optional offload of the cold history to system RAM
  (`--kv-window` / `--kv-host`) for long contexts.
- **Not in version 1:** images, video and audio models, image input to the LLM, other LLMs, and creating packs.

| Model | Type | Status |
|---|---|---|
| **Ripple-65B-Preview** | LLM (text) | **Version 1: supported** |
| Qwen3.8-27B | LLM (text) | Planned |

## Requirements

- An NVIDIA GPU with about 62 GB of GPU memory for the qvb8 1e-3 pack (about 50 GB for the qvb8r 1e-3 pack).
- **Provided build:** NVIDIA DGX Spark (GB10, sm_121, aarch64), Ubuntu 24.04, driver 580, CUDA 13.0, Python 3.12, PyTorch 2.14.
  This is the build that has been tested. Its prebuilt CUDA kernels are bundled in the tree
  (`src/qv_band/.prebuilt/sm_121a/`), so a GB10 needs no CUDA toolkit: download, install the packages, run.
- **V100:** prebuilt kernels are bundled for it as well (`src/qv_band/.prebuilt/sm_70/`, built with PyTorch 2.14.1
  for CUDA 12.6): a V100 with that PyTorch also runs without a CUDA toolkit.
- **Other GPUs:** the source is not tied to one GPU. The kernels compile for your GPU on first use (for example sm_120 on an
  RTX PRO 6000; a few minutes, the CUDA toolkit with `nvcc` is required), but only the DGX Spark build is tested for Ripple.
- **Compute dtype, `--dtype bf16|fp16`:** the server launcher passes `--dtype fp16`; add `--dtype bf16` after it to
  run in bf16, for example `scripts/ripple_server.sh start --dtype bf16`. Nothing
  is chosen automatically. A GPU without bf16 support (for example the V100, sm_70) emulates bf16 and is slow, so run it
  in fp16. On a GPU with bf16 (A100, H100, RTX 30/40/50, DGX Spark) both work: test which is faster on your card. The
  packs below are certified for both fp16 and bf16, so the error of every restored weight is less than tau in either
  dtype. fp16 tops out at 65504, so an activation can overflow in fp16 where bf16 would not; compare a few answers
  against bf16 if you can.
- **`escape_tables.json` (needed for fp16):** each pack folder on Hugging Face carries this small file next to
  `manifest.json`. It holds the fp16 escape table: the few weights an fp16 decode must take from the stored originals.
  An fp16 run without it stops at start-up and tells you to download it; a bf16 run does not need it.
  For the V100 the code has its own kernel paths (register-blocked GEMV, WMMA attention) and needs CUDA 12.x, because
  CUDA 13 dropped sm_70. The V100 path is not yet verified on Ripple.
- Python 3.10+ (tested 3.12) and the packages in `requirements.txt` (tested versions pinned).
- A Ripple target pack and a DFlash2 drafter pack (each pack dir carries its own config, modeling code and tokenizer).

## Weight packs (Hugging Face)

The packs are published separately on Hugging Face, from [huggingface.co/addiesam](https://huggingface.co/addiesam).
Download one target pack and one matching draft pack:

| Pack | Format | tau | Size | Download |
|---|---|---|---|---|
| Ripple-65B-Preview | qvb8 | 1e-3 | 57 GB | [Ripple-65B-Preview-QVB8-1e-3](https://huggingface.co/addiesam/Ripple-65B-Preview-QVB8-1e-3) |
| Ripple-65B-Preview | qvb8r | 1e-3 | 47 GB | [Ripple-65B-Preview-QVB8R-1e-3](https://huggingface.co/addiesam/Ripple-65B-Preview-QVB8R-1e-3) |
| Ripple-65B-Preview | qvb8 | 2e-3 | 49 GB | [Ripple-65B-Preview-QVB8-2e-3](https://huggingface.co/addiesam/Ripple-65B-Preview-QVB8-2e-3) |
| Ripple-65B-Preview | qvb8r | 2e-3 | 38 GB | [Ripple-65B-Preview-QVB8R-2e-3](https://huggingface.co/addiesam/Ripple-65B-Preview-QVB8R-2e-3) |
| Ripple-65B-Preview | qvb8r | 2.5e-3 | 35.5 GB | [Ripple-65B-Preview-QVB8R-2.5e-3](https://huggingface.co/addiesam/Ripple-65B-Preview-QVB8R-2.5e-3) |
| DFlash2 draft model | qvb8 | 5e-3 | 4.1 GB | [Ripple-65B-Preview-DFlash2-QVB8-5e-3](https://huggingface.co/addiesam/Ripple-65B-Preview-DFlash2-QVB8-5e-3) |
| DFlash2 draft model | qvb8r | 5e-3 | 3.4 GB | [Ripple-65B-Preview-DFlash2-QVB8R-5e-3](https://huggingface.co/addiesam/Ripple-65B-Preview-DFlash2-QVB8R-5e-3) |

```bash
pip install -U huggingface_hub
hf download addiesam/Ripple-65B-Preview-QVB8-1e-3 --local-dir ./models/Ripple-65B-Preview-QVB8-1e-3
hf download addiesam/Ripple-65B-Preview-DFlash2-QVB8-5e-3 --local-dir ./models/Ripple-65B-Preview-DFlash2-QVB8-5e-3
```

## Install

```bash
git clone https://github.com/addiesam/qvb8.git && cd qvb8
python -m venv .venv && . .venv/bin/activate
# 1. PyTorch with CUDA support for your platform (see pytorch.org). Tested: 2.14.0 built for CUDA 13.0.
# 2. The other packages:
pip install -r requirements.txt
# 3. Optional: flash-attn (faster attention). Without it the engine falls back to PyTorch attention.
#    It must be built against your PyTorch:  pip install flash-attn==2.8.3.post1 --no-build-isolation
```

On a GB10 (or a V100 with the matching PyTorch) the bundled prebuilt kernels are used as-is: no compile, no CUDA
toolkit. On any other GPU the CUDA kernels compile automatically on first use (a few minutes, `nvcc` required) into
`.ext_cache/` and are reused after that.


<div align="center">

# ❤️ SUPPORT THIS WORK ❤️

## I am an independent researcher and I need machines to test on.
## Every sponsor helps me publish more models, for free.

# 👉 [**SPONSOR DR. ADDIE SAM ON GITHUB**](https://github.com/sponsors/addiesam) 👈

[![Sponsor](https://img.shields.io/badge/SPONSOR-GitHub%20Sponsors-ea4aaa?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/addiesam)

</div>


## Run the server

```bash
export MODEL=./models/Ripple-65B-Preview-QVB8-1e-3
export DRAFT=./models/Ripple-65B-Preview-DFlash2-QVB8-5e-3
scripts/ripple_server.sh            # start in the background, wait until ready
scripts/ripple_server.sh status
scripts/ripple_server.sh log        # follow scripts/logs/Ripple-65B.log
scripts/ripple_server.sh startup    # the load / placement / kernel / memory lines from this start
scripts/ripple_server.sh stop
scripts/ripple_server.sh restart --no-think    # extra flags go to the server and win
```

More than one card in one machine: add `--pipeline cuda:0,cuda:1` (the cards, in layer order; the last is the main
card). One process; each card holds a contiguous range of layers, so a pack larger than one card's memory loads.
Several machines: run `python -m ai_engine.cli.stage_worker $MODEL --stage-port 50818 --stage-layers 0:12` on the
other box and add `--pipeline host:50818,local --pipeline-split 12` here (the link is unauthenticated: use a private
network).

Default port is 8080 (override with `PORT=...`). The server speaks the OpenAI chat API:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a haiku about rivers"}],"max_tokens":200,"stream":false}'
```

It supports streaming, tool calls (JSON) and the reasoning (`<think>`) channel. It also accepts content-part lists,
as sent by Cline and similar clients.

## Chat in the terminal

The chat is a client for a running server (it loads no model; standard library only):

```bash
scripts/ripple_chat.sh                                 # interactive, server on port 8080
scripts/ripple_chat.sh "Explain entropy" --no-interactive
PORT=9090 scripts/ripple_chat.sh                       # a server on another port
URL=http://host:8080 scripts/ripple_chat.sh --no-think  # a server on another machine
```

In the chat: `/reset` starts a new conversation, `/exit` (or Ctrl-D) leaves, Ctrl-C stops the current answer.

## Supported options

The release supports GPU only. These are the supported settings (the defaults are in the launcher):

- `--kv-cache qvb8` (certified compressed KV) or `bf16`
- `--prefill-linear qvb8`
- `--drafter dflash`, `ngram` or none
- `--pipeline cuda:0,cuda:1,...` (several cards in one machine) and `--pipeline-split`
- `--kv-window N|auto`, `--kv-host`, `--kv-cold-block T`, `--kv-gpu-reserve-mb M` (KV cache offload to system RAM)

Other values that the CLI help still lists, such as `--prefill-linear int8`, CPU or MTP drafters, are not part of
this release.

## Performance (DGX Spark, qvb8 1e-3 + 5e-3 drafter, greedy)

| Workload | Speed |
|---|---|
| Decode, code | ~15.8 tok/s |
| Decode, prose | ~4.5 tok/s |
| Two concurrent requests | ~18.7 tok/s aggregate |
| Prefill, 75k-token prompt | ~386 tok/s |

Speculative decoding is verified by the full model. In our tests, greedy output matched plain decoding; where two tokens are an
exact near-tie, the choice can differ.

## Update history

**v1.1.0**
- The engine is updated to the current development code: the model runs through the engine adapter (the Ripple
  adapter is registered by name), the continuous-batching scheduler is the current one, and the KV cache gains the
  sliding-window layers and the certified per-channel codec of the current engine.
- Several GPUs in one machine: `--pipeline cuda:0,cuda:1` splits the layers across the cards in one process
  (placement by layer range; the last card is the main card). This replaces tensor parallelism, which is no longer
  in the engine.
- Several machines: a remote stage worker (`python -m ai_engine.cli.stage_worker`) runs a layer range on another
  box; the server reaches it with `--pipeline host:port,local --pipeline-split N`.
- KV cache offload to system RAM for long contexts: `--kv-window N|auto` keeps the last N tokens (or an automatic
  window) in VRAM and the cold history in pinned host RAM; `--kv-host` moves the whole cache to host RAM.
- The terminal chat is now a client for a running server (`scripts/ripple_chat.sh`, standard library only): it sends
  OpenAI-style requests and prints the answer; it loads no model and works from any machine that can reach the
  server.
- The server launcher passes `--dtype fp16` (measured faster on the DGX Spark) and `--kv-tau-threshold 0.2`; add
  `--dtype bf16` to run in bf16.
- The launcher keeps a startup log (`scripts/logs/Ripple-65B.startup.log`, action `startup`): the command line and
  everything the server printed until it answered the health check -- the load, placement, kernel and memory lines,
  apart from the request log.
- Prebuilt CUDA kernels are bundled in the tree (`src/qv_band/.prebuilt/`): sm_121a for the DGX Spark (PyTorch
  2.14.0 for CUDA 13.0) and sm_70 for the V100 (PyTorch 2.14.1 for CUDA 12.6). A machine with the matching PyTorch
  runs without a CUDA toolkit; the sidecar next to each kernel records the sources, torch and platform it was built
  for, and anything else compiles on first use as before. The sidecar match no longer binds to the torch install
  path.

**v1.0.1**
- One switch for the compute dtype: `--dtype bf16|fp16` (replaces the `QV_MODEL_DTYPE` variable). The launchers set
  the default; nothing is chosen automatically.
- Two escape tables: an fp16 run reads only the fp16 table from `escape_tables.json`; a bf16 run reads the pack's full
  list. The pack files are unchanged. In our test (QVB8-1e-3, fp16, DGX Spark, one prompt, greedy) decoding went from
  about 1.8 to about 5.7 tokens per second; bf16 is unchanged.
- An fp16 run needs `escape_tables.json` in the pack folder; without it, the engine stops with an error at start-up.
- The server log: a client that hangs up before the reply (for example a start-up health check) is one `[http] ...` line, not a traceback; an fp16 run no longer prints the bf16 GEMM count as if it were in use.

**v1.0.0: First release**
- QVB8 Inference Engine for Ripple-65B-Preview, with the QVB8 and QVB8R weight packs.

## Papers

The certified compression method (every restored weight within error bound τ of the original) and its use on language models
are described in:

> Sam, Chi Ngai. *Certified L∞-Bounded Compression for AI Models, I: The Contract and the Virtual-Band Index.*
> Zenodo, 2026. [doi:10.5281/zenodo.23815007](https://doi.org/10.5281/zenodo.23815007)

> Sam, Chi Ngai. *Certified L∞-Bounded Compression for AI Models, II: Compressing a Language Model's Weights, Key–Value Cache and Activations.*
> Zenodo, 2026. [doi:10.5281/zenodo.23816044](https://doi.org/10.5281/zenodo.23816044)

## License

The code is © 2026 Sam, Chi Ngai, licensed under [Apache-2.0](LICENSE) (see [NOTICE](NOTICE)).
Questions and bug reports: please open a [GitHub issue](https://github.com/addiesam/qvb8/issues).

The paper is licensed under CC BY-NC-ND 4.0.
The model weights are not part of this repository and keep their own license.

## Support this project

- **Individuals:** if this project is useful to you, please consider
  [sponsoring Dr. Addie Sam on GitHub](https://github.com/sponsors/addiesam).
- **Companies:** for custom model packs (other models, other GPUs) or support, open a GitHub issue or contact the author through [github.com/addiesam](https://github.com/addiesam).

## Disclaimer — no warranty, no liability

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
