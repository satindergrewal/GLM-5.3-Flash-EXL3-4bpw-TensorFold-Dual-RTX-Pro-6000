<h1 align="center">GLM-5.3-Flash EXL3 on 2x RTX PRO 6000 with TensorFold</h1>

<p align="center"><sub>by <a href="https://github.com/satindergrewal">Satinder Grewal</a> · built on <a href="https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold">Mia's AI Lab's DGX Spark recipe</a> · mixed-rate EXL3 support in <a href="https://github.com/satindergrewal/TensorFold/tree/mixed-k34">a TensorFold fork</a></sub></p>

<p align="center">
  <a href="https://github.com/satindergrewal/TensorFold/tree/mixed-k34"><img src="https://img.shields.io/badge/TensorFold-v0.6.5_(fork)-A9D5CE?style=for-the-badge&amp;labelColor=151615" alt="TensorFold v0.6.5 fork"></a>
  <img src="https://img.shields.io/badge/GPUs-2x_RTX_PRO_6000-EEB07E?style=for-the-badge&amp;labelColor=151615" alt="2x RTX PRO 6000">
  <img src="https://img.shields.io/badge/Quants-4bpw_+3.5bpw_mixed-D99288?style=for-the-badge&amp;labelColor=151615" alt="4bpw and 3.5bpw mixed">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-D99288?style=for-the-badge&amp;labelColor=151615" alt="Apache-2.0"></a>
</p>

Serve **GLM-5.3-Flash** from one Linux host with **two 96 GB RTX PRO 6000 Blackwell GPUs**, through an
OpenAI-compatible API, with **4 concurrent requests**, the model's full **~1M native context**, **image input**,
DFlash2 speculative decoding, tool calling, structured outputs, `/tokenize` and Prometheus `/metrics` — in **two
switchable quant arms**: the unmodified stock 4bpw checkpoint, or Mia's AI Lab's mixed 3.5bpw encode. One command
per arm starts it.

- **Checkpoints.** 4bpw arm: [`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw),
  served from the byte-identical
  [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) mirror.
  3.5bpw arm: the companion [k35 mixed encode](https://github.com/satindergrewal/GLM-5.3-Flash-EXL3-3.5bpw-Mixed-SM120-TP2)'s
  artifact — EXL3 routed experts at mixed k3/k4 per-tensor rates (~146 GB), built to bring 1M-context GLM inside
  192 GB under vLLM; TensorFold holds the stock 4bpw at the same window, and this recipe's fork carries the
  mixed-rate loader so both arms run on the same engine.
- **Drafter:** [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
  DFlash2 is non-commercial (CC BY-NC-ND 4.0).
- **API model ids:** `GLM-5.3-Flash-EXL3-4bpw` and `GLM-5.3-Flash-EXL3-3.5bpw-mixed`.
- **Context:** 1,048,576 tokens (4bpw) / 1,048,560 (3.5bpw), sharing an FP8 KV pool.
- **Concurrency:** 4 requests by default, with DFlash2 drafting and prompt-chunk prefill while others decode.
- **Vision:** up to 128 images per request (`TENSORFOLD_GLM_MAX_IMAGES`), both arms.

## Performance

Two RTX PRO 6000 Blackwell 96 GB GPUs at stock clocks, one host, driver 580.178.04. Decode tok/s are engine
deltas (reasoning + content) over the decode window; aggregate is completion tokens divided by wall time for the
batch. Greedy (temperature 0) unless noted. Numbers without a second date are the 2026-10-02 (4bpw) and
2026-10-04 (3.5bpw, TensorFold 0.6.5 merge) measurements below; the two arms' protocols differ where noted.

**Decode** (single stream and 4 concurrent streams; thinking on)

| Concurrent requests | TensorFold 4bpw | TensorFold 3.5bpw mixed | vLLM 3.5bpw (reference) |
| ---: | ---: | ---: | ---: |
| 1 stream | 63.2 tok/s | **166.4 prose / 208.3 JSON** (release config) | 143 tok/s |
| 4 streams, prose | 325.1 tok/s | **308–360 tok/s** (release config) | 141.2 tok/s |
| 4 streams, JSON | 375.2 tok/s | not re-run post-fix | not published |

\* the 3.5bpw 4-stream run used a shorter-generation protocol than the 4bpw's (early end-of-sequence);
a like-for-like re-measure is pending. TTFT at a ~2-3k prompt: 0.74 s (4bpw) / 1.78 s (3.5bpw).

![Single-stream decode](charts/decode-single.svg)

![Aggregate decode at 4 streams](charts/decode-concurrency.svg)

**Prefill** (TTFT-derived; methods differ per row and are stated)

| Prompt | TensorFold 4bpw | TensorFold 3.5bpw mixed | vLLM 3.5bpw (reference) |
| --- | ---: | ---: | ---: |
| ~2-150k tokens | ~3.5k tok/s @128k | ~2.1k tok/s @166k (TTFT 78.3 s, speed config) | 2,793-2,841 tok/s @500-950K |
| ~1M tokens | 2,379 tok/s effective @1.008M | ~1.67k tok/s effective @912k | not published |

The 3.5bpw arm prefills in 1024-row chunks against the 4bpw path's 2048 (a kernel shared-memory ceiling);
restoring the 2048-row chunks through the per-expert-width path is the known next lever.

![Prefill throughput](charts/prefill.svg)

**Prompt reuse** (cold first time versus warm identical re-send)

| Prompt | TensorFold 4bpw | TensorFold 3.5bpw mixed |
| --- | ---: | ---: |
| ~64-70k tokens, first time | 54.3 s | 39.7 s |
| Same prompt again | **0.27 s (~200x)** | **0.13 s (~300x)** |

**Long context** (needle retrieval, drafting on)

| Prompt | Depth | Result |
| --- | ---: | --- |
| 826,051 tokens (4bpw) | 87% | retrieved exactly |
| 1,008,051 tokens (4bpw) | 90% | retrieved exactly (423.6 s wall; peak VRAM 92.0 / 90.8 GiB of 95.6) |
| ~912,000 tokens (3.5bpw) | 90% | retrieved exactly (546.6 s wall on the 0.6.5 merge; 23.3 min pre-merge) |

![Max served context](charts/context-by-stack.svg)

**Quality** (250-problem GSM8K test slices, greedy)

| Benchmark | TensorFold 4bpw | TensorFold 3.5bpw mixed | vLLM 3.5bpw (same box) |
| --- | ---: | ---: | ---: |
| GSM8K | 97.2% | **98.4% (246/250)** | 96.89% |
| DFlash2 draft acceptance | 73.9% | 74.9% | MTP3 / DFlash2 (their protocol) |
| Vision, color-ID probe | pass | pass | — |

The mixed 3.5bpw encode carries a five-run KLD gate against its teacher from encode time (mean 0.0246,
bar 0.06). Same hardware for every number above; the Spark reference for the mixed checkpoint's own recipe
is 98.0% at the same slice (Mia's AI Lab).

![GSM8K](charts/gsm8k.svg)

## Requirements

- **Linux x86_64**, two **96 GB RTX PRO 6000 Blackwell** GPUs, NVIDIA drivers with CUDA forward compatibility
  (the container runs CUDA 13.3 userspace; validated on driver 580.178.04).
- **Docker** with the NVIDIA container toolkit and access to the NVIDIA container registry.
- **Disk:** ~180 GiB for the 4bpw checkpoint or ~146 GiB for the 3.5bpw artifact, ~2.3 GiB for DFlash2,
  ~25 GiB for the image, plus kernel-cache space.
- Bash, Python 3, `git`, `curl`. First build needs GitHub, NVIDIA's registry and package indexes.
- Optional: a Hugging Face token for downloads.

## Quick start

```bash
git clone https://github.com/satindergrewal/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Dual-RTX-Pro-6000.git
cd GLM-5.3-Flash-EXL3-4bpw-TensorFold-Dual-RTX-Pro-6000
cp .env.example .env                 # set MODEL_DIR_4BPW / MODEL_DIR_35BPW and QUANT
./build-image.sh                     # TensorFold from the pinned fork + the recipe's patches
./download-dflash2.sh                # the drafter (~2.3 GB)

QUANT=3.5bpw ./serve/serve-tf.sh     # or QUANT=4bpw ./serve/serve-tf.sh
```

The launcher starts one container with two TensorFold ranks (one per GPU, NCCL over PCIe), waits for the API
and prints the endpoint. The API base is `http://127.0.0.1:8888/v1`.

```bash
curl -s http://127.0.0.1:8888/v1/models
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3-3.5bpw-mixed",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 2000
}'

./stop.sh                               # stop the ranks
# any setting change: re-run the launcher - it recreates the container
QUANT=3.5bpw ./serve/serve-tf.sh
./status.sh                              # container and lane state
docker logs -f glm53-flash-tf-box        # rank logs
curl -s http://127.0.0.1:8888/health     # busy flag, streams, pool tokens, draft counters
```

An export wins over `.env` (caller precedence): `QUANT=4bpw ./serve/serve-tf.sh` overrides a `.env` that says
3.5bpw. The model replies with thinking in `reasoning_content` and the answer in `content`, so give requests
enough `max_tokens`.

## Images

Vision is on for both arms (`VISION=1`). Send images as OpenAI-style `image_url` content parts in a user
message; the tower runs on rank 0 (1.05 GiB of bf16 weights):

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3-3.5bpw-mixed",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
    {"type": "text", "text": "What is in this picture?"}]}],
  "max_tokens": 2000
}'
```

Up to 128 images per request (`TENSORFOLD_GLM_MAX_IMAGES`), at most 2,048 tokens a picture. The 3.5bpw
artifact ships the image-capable chat template (its own k35-native template has no branch TensorFold's
`--vision` can extend; the original is kept as `chat_template.jinja.k35-orig` in the artifact).

## Configuration

Settings come from the **environment**, then **`.env`**, then the defaults in
[`serve/serve-tf.sh`](serve/serve-tf.sh), in that order.

| Setting | Default | Meaning |
| --- | --- | --- |
| `QUANT` | `4bpw` | `4bpw` or `3.5bpw`: model directory, served id and image per arm |
| `MODEL_DIR_4BPW` / `MODEL_DIR_35BPW` | from `.env` | Checkpoint paths for each arm |
| `TF_VERSION` / `IMAGE` | `v0.6.5-mixed-k34` | TensorFold build and image tag (`tensorfold-glm53:TAG`) |
| `PORT` / `MASTER_PORT` | `8888` / `29551` | API listener and NCCL master port |
| `PARALLEL` | `4` | Concurrent requests |
| `CONTEXT` | `0` (largest that fits) / `1048576` (3.5bpw) | Prompt plus reply window per request |
| `MAX_TOKENS` | `32768` | Default completion budget |
| `KV` | `fp8` | KV representation (`bf16` needs more memory) |
| `DENSE` | `q4` | Dense-layer format (`q4`, `fp8`, or checkpoint `bf16`) |
| `VISION` | `1` | Image input |
| `TENSORFOLD_GLM_MAX_IMAGES` | `128` | Images per request |
| `ABLIT` | `0` | Reserved: `1` will serve an abliterated checkpoint per arm (`MODEL_DIR_ABLIT_*BPW`, not wired yet) |
| `RANK0_GPU` / `RANK1_GPU` | `0` / `1` | GPU indices |

## The 3.5bpw mixed arm: what the fork changes

The mixed encode picks a bit rate per expert tensor (4,880 experts pure k4, 4,823 pure k3, 2,681
split-projection across gate/up/down). Upstream TensorFold's own loader refuses mixed-bit markers, so
[satindergrewal/TensorFold, branch `mixed-k34`](https://github.com/satindergrewal/TensorFold/tree/mixed-k34)
carries the support: the mixed marker accepted for glm5_next exl3+mcg, each expert tensor's width read from
its own trellis shape, mixed layers routed through the per-expert-width `Exl3RoutedExperts` ABI (uniform
layers keep the stacked path), and the routed pass slicing the pick to the contiguous top-k columns — the
group kernel indexes the pick with flat `r * slots + s` arithmetic, so a `[R, top_k + 1]` pick against a
top_k-slot scratch shifts every row's expert window by one position from row 1 on. That single alignment bug
was the whole "mixed arm is broken" story: it scored 36% on GSM8K-25 while producing fluent text until the
one-line contiguous-pick fix took it to 96%, and 98.4% at 250 problems.

**2026-10-04: upstream v0.6.5 merged into the fork** (`5cf0b3b` on `mixed-k34`), then **Aevonix
Research's full 32-patch single-host engine series landed** (branch
[`mixed-k34-avx`](https://github.com/satindergrewal/TensorFold/tree/mixed-k34-avx)): IPC all-gather with
protocols, the copy-engine exchange, two prefill lanes, launch tables, decode fusions, draft-fast, wide
windows, expert prompt kernels, prefill-2/3, round-cap two-shot, lane inputs/partials, warm-turn
incremental and the port series — every patch hand-merged onto this tree with the mixed-rate path
preserved and re-gated (GSM8K-250 98.4% unchanged on the integrated stack). The fused decode kernels and
the GPU sampler are **compiled in but default OFF** (`TF_GLM_FUSE=0`, `TENSORFOLD_GPU_SAMPLE=0`): the
hand-merged kernel sources fail their first-use bit-checks here, and TensorFold's checks catch it —
serving stays exact while they are off. Earlier state — the symmetric
two-rank exchange, the generic EXL3 route, API keys and the vLLM-metric mirrors came in; the mixed-rate
loader, the pick-stride fix and the fp8-KV stack were preserved and re-gated (GSM8K-250 98.4% unchanged;
the ~912k needle wall dropped 23.3 → 9.1 min). Upstream's own loader still declines mixed-bit markers, so
the mixed-rate loader remains fork-local; logprobs on the GLM-5.3 backend also remain unavailable upstream.

The decode-path kernels are shared with upstream; the mixed arm's prefill runs 1024-row chunks against the
stacked path's 2048 (a group-launch shared-memory ceiling) — the known next performance lever, measured in
the prefill table above.

## Quant fidelity

| | TensorFold 4bpw | TensorFold 3.5bpw mixed | vLLM 3.5bpw (same box) |
| --- | ---: | ---: | ---: |
| GSM8K, greedy, 250-slice | 97.2% | **98.4%** | 96.89% |

Same hardware: the 3.5bpw mixed encode under TensorFold is behaviorally at least the equal of stock 4bpw
under TensorFold and of itself under vLLM (all within ~1.5pt of slice noise), matching the encode's five-run
KLD gate against teacher (mean 0.0246, bar 0.06). Dense and KV quantization change numerics; the exactness
boundary for each intentional numerical change is the deploy-time gate set (GSM8K slices, needle recall,
vision probes, 13x17-style smoke checks) re-run on every engine or loader change.

## Repository layout

```text
serve/            launcher (serve-tf.sh), rank script, stop and status
patches/          the recipe's TensorFold patch series
charts/           the performance graphs above, generated by make_charts.py
bench/            the gate scripts (GSM8K slices, throughput matrix, vision probe)
docs/             memory-geometry notes
build-image.sh    build the serving image from the pinned fork + patches
download-dflash2.sh
publish-docker.sh
.env.example      every knob with defaults
LICENSE           Apache License 2.0
```

## License

This repository's code and documentation are **Apache-2.0**
([LICENSE](LICENSE)). Upstream code and model weights keep their own licenses. **DFlash2 weights are
CC BY-NC-ND 4.0: non-commercial, not redistributed here.** No model weights, private data or credentials
are included.

## Credits

[TensorFold](https://github.com/ashhart/TensorFold) is by **Ash Hart**. The **EXL3 checkpoints and the
original recipe** are by [Mia's AI Lab](https://huggingface.co/Mia-AiLab) — this recipe's structure follows
their [DGX Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) and
[Aevonix Research's](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold) single-host
presentation. **DFlash2** is by [Inco AI](https://huggingface.co/incoai). The mixed 3.5bpw encode is from the
companion [k35 mixed repo](https://github.com/satindergrewal/GLM-5.3-Flash-EXL3-3.5bpw-Mixed-SM120-TP2).
The dual-arm single-host recipe, the mixed-rate TensorFold support and the measurements are by
[Satinder Grewal](https://github.com/satindergrewal). 
