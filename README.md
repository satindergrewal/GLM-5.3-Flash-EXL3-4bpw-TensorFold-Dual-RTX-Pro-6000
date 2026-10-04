# GLM-5.3-Flash EXL3 - TensorFold v0.6.5 (fork) - Dual RTX Pro 6000 (4bpw stock + 3.5bpw mixed)

GLM-5.3-Flash on 2x RTX PRO 6000 (SM120, 96 GB each), single box, **two switchable
arms** (`QUANT=4bpw|3.5bpw`): the unmodified stock TR3-4bpw checkpoint, or the
MiaAi-Lab k3/k4 per-tensor mixed 3.5bpw encode — both with the full ~1M native
window, concurrent streams, vision, and DFlash2 speculative decoding.

The 4bpw arm is GLM-5.3-Flash EXL3 TR3-4bpw — quantized by MiaAi-Lab
([Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw),
served from the byte-identical
[brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
mirror). The 3.5bpw arm is the companion
[k35 mixed encode](https://github.com/satindergrewal/GLM-5.3-Flash-EXL3-3.5bpw-Mixed-SM120-TP2)'s
artifact. Both run on a fork of [TensorFold](https://github.com/ashhart/TensorFold) — upstream v0.6.5 merged with this recipe's mixed-rate support (branch `mixed-k34` of [satindergrewal/TensorFold](https://github.com/satindergrewal/TensorFold)) — in
`COMM=nccl` mode — the first (to our knowledge) published TensorFold recipe for
x86_64 discrete GPUs. The same 192 GB that needs a 3.5bpw mixed encode under
vLLM holds stock 4bpw + the full 1M window here, because TensorFold's runtime
carries no vLLM-style context-proportional workspace and repacks dense weights
to q4. Mixed-rate experts (k3/k4 per tensor) ride the fork's per-expert-width
path; stock 4bpw keeps the stacked fast path.

## Validated

| Check | Result |
|---|---|
| Window allocated (4bpw, 2026-10-02) | 1,048,576 tokens, both ranks (88.09 / 86.29 GiB within 90.08 GiB budgets) |
| Window allocated (3.5bpw, 2026-10-04) | 1,048,560 context served (`CONTEXT=1048576`), 77.4 / 79.2 GiB (vision on) — ~12.6 GiB spare |
| Long-context recall (4bpw) | 826,051-token prompt @87% and 1.008M @90%: **retrieved exactly** |
| Long-context recall (3.5bpw, 2026-10-04) | ~912k-token prompt @90% depth: **retrieved exactly**; v0.6.5 merge cut the wall 23.3 → 9.1 min |
| Health/geometry | rank 0 + rank 1 ready in ~209 s (warm caches); API healthy |
| Vision (4bpw) | tower loaded; image cap raised to 128/request (see patches) |
| Vision (3.5bpw, 2026-10-04) | **pass**: 3-image color ID at the 1M window; artifact ships the 4bpw chat template for TF's `--vision` |
| `/v1/models` | vLLM-compatible shape incl. `max_model_len` (recipe patch 0053-v1-models-context) |
| GSM8K accuracy | 4bpw 97.2%, 3.5bpw **98.4%** (250-problem slices, greedy — Results below) |
| Routing correctness | the 2026-10-03 pick-stride fix (fork) — before it the mixed arm scored 36%; see the 3.5bpw section |

**Testing status: partially validated.** Measured: window allocation on both
arms, needle recall to ~1M (4bpw) and ~912k (3.5bpw), decode/aggregate throughput
at 1-4 streams, TTFT at 2k/128k, prompt reuse, vision (2-image 4bpw, 3-image
3.5bpw), GSM8K-250 both arms, DFlash2 acceptance. **Not yet run:** HumanEval,
video input, real-photo vision, 128-image stress on the 3.5bpw arm, multi-day
agentic soak, safety red-team, NVFP4 comparisons, and a like-for-like re-measure
of the 3.5bpw aggregate (its current number used a shorter-generation protocol
than the 4bpw's — noted in the table).

## Results (2026-10-02, this exact stack, driver 580.178.04)

Policy: numbers without a date are unverified. Decode tok/s are engine deltas
(reasoning + content) over the decode window; aggregate = completion tokens /
wall for the batch. Greedy unless noted.

| Metric | Value | Notes |
|---|---|---|
| Decode, prose, 1 stream | **63.2 tok/s** | 2k prompt, 512-token reply |
| Decode, prose, 2 streams | 191.1 tok/s aggregate | 2x512 completion tokens / wall |
| Decode, prose, 4 streams | **325.1 tok/s aggregate** | 4x512 completion tokens / wall |
| Decode, structured (JSON), 1 stream | 58.3 tok/s | |
| Decode, structured (JSON), 4 streams | **375.2 tok/s aggregate** | |
| TTFT, ~2k prompt | 0.74 s | |
| TTFT, ~128k prompt | 36.9 s | ~3.5k tok/s prefill |
| Needle, 826,051-token prompt @87% depth | **retrieved exactly** | first try |
| Needle, 1,008,051-token prompt @90% depth | **retrieved exactly** | 423.6 s wall (~2.4k tok/s incl. prefill+decode); peak VRAM 92.0 / 90.8 GiB of 95.6 |
| Prompt reuse, 64,416-token prompt | pass 1: 54.3 s → pass 2: **0.27 s** (~200x) | kept within the window pool even with `TF_GLM_CACHE_GIB=0`; only the most recent conversation's state is kept |
| Vision, 2-image color ID | pass | correct order + colors |
| GSM8K, 250-problem test slice, greedy | **97.2% (243/250)**, 220 s | DFlash2 drafting on; Mia's Spark number at 250: 98.0% |
| GSM8K, 250-problem slice, QUANT=3.5bpw arm, greedy (2026-10-03) | **98.4% (246/250)** | post pick-stride fix, image v0.6.5-mixed-k34 (upstream 0.6.5 merged); full native window; GSM8K-25 on the same lane: 24/25 |
| DFlash2 draft acceptance | 73.9% (105,304 / 142,458 drafted) | across the whole benchmark session, 563 requests |

Measurement caveats, stated plainly: decode streams counted engine deltas
(reasoning + content ≈ 1 token each); TTFT is first *token* delta, not first
byte; the 1.008M needle is a synthetic needle, not an MRCBench-style
multi-needle suite; GSM8K is a 250-problem slice of the 1,319-problem test set,
scored by `#### <number>` extraction (one benign parser iteration was needed —
the first run's 0.0% was a scorer bug comparing against the full gold
annotation string, retracted).

## Head to head on this box

Same hardware (2x RTX PRO 6000, 192 GB), different engines and quants. Sources:
vLLM 3.5bpw mixed — companion repo README (1m-multi profile unless noted);
vLLM K4 4bpw — `v84` release validation on this box (98k context,
nvfp4_ds_mla KV, DFlash2 mean acceptance 5.74 of 7, no throughput published);
TensorFold — Results above.

![Max served context](charts/context-by-stack.svg)

| | vLLM K4 4bpw (v84) | vLLM 3.5bpw mixed (1m-multi) | TensorFold 3.5bpw mixed (QUANT=3.5bpw) | TensorFold 4bpw (this recipe) |
|---|---|---|---|---|
| Max context | 98,304 | 1,000,000 | **1,048,560** (needle-verified at ~912k @90% depth) | **1,048,576** |
| Weights | EXL3 K4 4bpw | EXL3 mixed 3.5bpw | EXL3 mixed 3.5bpw | **stock EXL3 TR3 4bpw** |
| KV cache | nvfp4_ds_mla | fp8_ds_mla / calibrated NVFP4 MLA | fp8 (e4m3 latent + indexer) | fp8 (e4m3 latent + indexer) |
| Decode single (thinking on) | not published | 143 tok/s | 81.0 tok/s (v0.6.5) | 63.2 tok/s |
| Aggregate @4 streams | not published | 141.2 tok/s | 113.6 tok/s prose on v0.6.5 (early-EOS shortened generations) | **325.1 prose / 375.2 JSON** |
| Prefill | not published | 2,793–2,841 tok/s @500–950K | ~1.9k tok/s @~150k (TTFT 80.7 s); ~1.67k effective @912k (v0.6.5) | ~3.5k @128k; ~2.4k effective @1.008M |
| TTFT, ~2–3k prompt | not published | not comparable | 1.78 s | 0.74 s |
| Prompt reuse, ~70k prompt | not published | — | 39.7 s cold → **0.13 s warm (~300x)** | 54.3 s → 0.27 s (~200x) |
| Needle, ~912k-token prompt @90% depth | not published | — | **retrieved exactly**, 9.1 min wall on v0.6.5 (~1.67k tok/s effective incl. prefill; 23.3 min on the pre-merge stack) | 826k @87% and 1.008M @90%: retrieved exactly |
| GSM8K (greedy) | not published | 96.89 | **98.4%** (250-slice) | **97.2%** (250-slice) |
| Images per request | vision smoke pass | 4 | **pass** (3-image color ID; artifact ships the 4bpw chat template for TF's `--vision`; 2026-10-04) | **128** |
| Drafting | DFlash2, 5.74/7 mean accept | MTP3 (~2.4 mean); DFlash2 3.6–4.2 accept | DFlash2, **74.9%** accepted (2026-10-04 boot counters) | DFlash2, 73.9% accepted |

![Single-stream decode](charts/decode-single.svg)

![Aggregate decode at 4 streams](charts/decode-concurrency.svg)

![Prefill throughput](charts/prefill.svg)

![GSM8K](charts/gsm8k.svg)

Read plainly, both directions: **vLLM's tuned 3.5bpw lane decodes a single
stream ~2.3x faster** (143 vs 63.2 tok/s thinking-on — different prompt
protocols, but the gap is real and its 174–178 tok/s thinking-off single
widens it), **while TensorFold holds the stock unmodified 4bpw at the full
native window and wins 4-stream aggregate by ~2.3x** (325 vs 141 tok/s). The
K4 vLLM validation on this box stopped at 98k context: 4bpw + 1M under vLLM is
the combination that does not fit, which is the gap this recipe closes by
switching engines instead of switching quants.

## Quant switch: 4bpw or 3.5bpw

`QUANT=4bpw|3.5bpw` in `.env` switches `MODEL_DIR` and the served model id
(`GLM-5.3-Flash-EXL3-4bpw` / `GLM-5.3-Flash-EXL3-3.5bpw-mixed`). An exported
`QUANT` from the caller wins over `.env` (precedence-preserving source).
Everything else — DFlash2, vision, window fit — is quant-agnostic. Both arms
serve from the merged image `tensorfold-glm53:v0.6.5-mixed-k34` (TensorFold
0.6.5 + the fork's mixed-rate support incl. the 2026-10-03 pick-stride fix).

### 3.5bpw under TensorFold: WORKING (2026-10-03)

The 3.5bpw mixed artifact boots and scores **98.4% GSM8K-250 (246/250, greedy)** under
TensorFold — above the same artifact's own vLLM number (96.89%) on the same box and slice.
It serves the **full native window** (1,048,560 context; measured 77.4 of 90.08 GiB/GPU at boot). One quirk: with `CONTEXT=0` (auto-fit) this arm stops one 64k block short (983,040) while the 4bpw arm's auto-fit reaches native, so the recipe pins `CONTEXT=1048576` for the 3.5bpw arm. The port chain that got here (fork branch
[`mixed-k34`](https://github.com/satindergrewal/TensorFold/tree/mixed-k34), v0.6.0 + the
MiaAI-Lab patch stack + the mixed-bits commits):

1. **Config parse** — `"bits": "mixed_k34_per_tensor"` hit
   `bits=int(quant.get("bits", 4))` (`weights.py:96`). Fixed: non-int bits fall back to 4
   for geometry; the trellis tensors carry their own widths.
2. **Family gate** — `glm5_next/__init__.py` validated against the uniform `EXL3_VARIANT`.
   Fixed: the mixed k3/k4 mcg variant is accepted (allowed_bits [3, 4]).
3. **Trellis width check** — `exl3_mm.py words()` demanded `int16 [..., 64]`. Relaxed:
   3-bit `[..., 48]` accepted.
4. **Per-expert-width MoE** — `moe_exl3` routes the mixed layer through
   `x3experts.prepare`/`x3experts.routed` (one width per expert, no uniform stack), with the
   per-slot `[R, top_k, hidden]` ey write (the earlier flat write caused repetition loops).

**The accuracy bug that remained after the port (36% GSM8K-25), found 2026-10-03 — one
line.** The x3 expert kernels (`group_kernel`, `rot_in_kernel`, `gateup_epilogue_kernel`,
`down_epilogue_kernel`, `down_combine_kernel` in `tensorfold/cuda/exl3/experts.cu`) index
the pick tensor with flat `r * slots + s` arithmetic, assuming a contiguous
`[R, slots=8]` layout. The serving path passes `b.pick[:R]`, which is `[R, top_k + 1]` —
stride 9, because slot 8 holds the shared expert. From row 1 onward every row's expert
window was shifted by one position: each token lost its 8th expert, inherited the previous
row's picks, and wasted a slot on the shared-expert id. Fluent text, degraded reasoning —
36% GSM8K-25. The fix: `b.pick[:R, :c.top_k].contiguous()` in the mixed branch of
`forward.py` (image `tensorfold-glm53:v0.6.1-mixed-k34`). GSM8K-25 went **36% → 96% (24/25)**
on restart, GSM8K-250 **98.4% (246/250)**.

Decode parity, sealed before the routing fix (this is why the quant itself was never
suspect after 2026-10-03): the sealed pure-torch B12X reader, TensorFold's python MCG
reference, and TensorFold's CUDA lane decode produce **bit-identical** rotated-domain
weights on real artifact tensors — both rates, all three projections (both tile geometries),
max abs diff 0.0; final weights with Hadamard + scales agree to 3e-15 (fp64 noise). The
mixed encode's mcg codebook (multiplier 0xCBAC1FED), 256-state lane permutation, cyclic
bit-stream window, and suh/svh scale convention are identical between the B12X/vLLM
convention and TensorFold's kernels.

Artifact fact (drove the per-expert-width design, still true): of 12,384 experts,
4,880 are pure k4, 4,823 pure k3, and **2,681 are split-projection** (gate/up/down at
different rates). The x3experts path takes widths per expert-projection tensor natively,
so no per-rate sub-launching is needed. The 4bpw arm (all-k4, stacked path) is unaffected
and serves the full 1M window at measured 97.2% GSM8K.

**2026-10-04: upstream v0.6.5 merged into the fork** ([satindergrewal/TensorFold, branch
`mixed-k34`](https://github.com/satindergrewal/TensorFold/tree/mixed-k34), commit `5cf0b3b`) —
the symmetric two-rank exchange, the generic EXL3 route, API keys and the vLLM-metric mirrors
came in; the mixed-rate support, the pick-stride fix and the fp8-KV stack were preserved and
re-gated (GSM8K-250 98.4% unchanged; the ~912k needle wall dropped 23.3 → 9.1 min). Upstream
still declines mixed-bit rates in its own loader, so the mixed-rate loader stays fork-local.
Logprobs on the GLM-5.3 backend also remain unavailable (upstream ships none).

### Fidelity of the two quants, measured behaviorally

Full-vocab KLD is not measurable on this stack: TensorFold implements **no
logprobs** (chat completions with `logprobs: true` → 400 "logprobs are not
supported by this model or backend"), so the teacher-forcing KLD methodology
has no student-side signal. The behavioral proxy that does work, on this box:

| | TensorFold 4bpw (this recipe) | TensorFold 3.5bpw mixed (QUANT=3.5bpw) | vLLM 3.5bpw mixed (companion repo, 700k–1M lanes) |
|---|---|---|---|
| GSM8K, greedy, 250-slice | **97.2%** | **98.4%** | 96.89% |

Same hardware: the 3.5bpw mixed encode under TensorFold is behaviorally at least the equal
of stock 4bpw under TensorFold and of itself under vLLM (all within ~1.5pt of slice noise).
That matches the 3.5bpw repo's five-run KLD gate vs teacher (mean 0.0246, bar 0.06)
measured during its encode.

## Hardware

| | |
|---|---|
| GPUs | 2x NVIDIA RTX PRO 6000 Blackwell 96 GB (validated on a mixed Workstation + Max-Q pair, PCIe) |
| System | x86_64 Linux, 125 GB RAM, driver 580.178.04 (container runs CUDA 13.3 userspace via forward compatibility) |
| Software | Docker + NVIDIA Container Toolkit (CDI), ~35 GB disk for the image, ~172 GB for the checkpoint |
| Network | none between ranks beyond the host loopback — `COMM=nccl`, no InfiniBand/RoCE needed |

## Quick start

```bash
./download-dflash2.sh     # DFlash2 draft (2.2 GiB, pinned revision) into ./hf
./build-image.sh          # tensorfold-glm53:v0.6.0 (base image + TensorFold + 54 patches)
./serve/serve-tf.sh       # both ranks, waits for nothing; poll /health
./stop.sh                 # remove the container
```

Then:

```bash
curl -s http://127.0.0.1:8888/health
curl -s http://127.0.0.1:8888/v1/models | jq '.data[0].max_model_len'   # 1048560
```

Boot is ~3.5 min with warm kernel caches; the first boot compiles the CUDA
extensions per GPU (count ~20-40 min). Point `MODEL_DIR` at any local copy of
the TR3-4bpw checkpoint (flat snapshot layout, 120 shards + config.json).

Every knob is an env override — see [.env.example](.env.example).

## What runs (the stack, layer by layer)

| Layer | What | Where |
|---|---|---|
| API | OpenAI-compatible (`/v1/chat/completions`, tools, reasoning, vision) | TensorFold rank 0, port 8888 |
| Engine | TensorFold 0.6.0, CUDA lane, TP2 | two ranks, one container |
| Model | GLM-5.3-Flash 320B MoE, stock TR3-4bpw EXL3 (routed experts 4bpw, BF16 elsewhere) | `/model` mount |
| Dense repack | q4 groups-of-64, head FP8, kv_b BF16 (`TF_GLM_DENSE=q4`) | at load |
| KV cache | fp8 (e4m3 + power-of-two scales), 1,048,576-token shared pool | `TF_GLM_KV=fp8` |
| Spec decode | DFlash2, pinned `bf582e4e` | `--drafter` |
| Parallelism | TP2, 4 decode streams, single box | `--tp 2 --parallel 4` |
| Comm | NCCL over PCIe, `COMM=nccl` (no RoCE) | `TF_GLM_COMM` |
| Context | 1,048,576 window (auto-fit, `--context 0`) | enforced limit 1,048,560 |

## Parameters

| Env | Default | Meaning |
|---|---|---|
| `QUANT` | 4bpw | quant switch: `4bpw` (stock TR3-4bpw mirror) or `3.5bpw` (mixed k3/k4 encode — **blocked, see below**) |
| `MODEL_DIR_4BPW` / `MODEL_DIR_35BPW` | (required per arm) | checkpoint directories for each arm |
| `MODEL_DIR` | (required) | local TR3-4bpw checkpoint directory (flat snapshot) |
| `PORT` / `MASTER_PORT` | 8888 / 29551 | API port / rank rendezvous (loopback) |
| `PARALLEL` | 4 | concurrent decode streams (1-4 with DFlash2) |
| `CONTEXT` | 0 (auto-fit) | pin a window (e.g. 728883) to free memory for other things |
| `MAX_TOKENS` | 32768 | reply budget when the client sends none |
| `TF_GLM_KV` | fp8 | `fp8` (1M window) or `bf16` (exact, ~196k window) |
| `TF_GLM_DENSE` | q4 | dense-weight repack: `q4` / `fp8` / `bf16` |
| `TF_GLM_COMM` | nccl | `nccl` (PCIe) or `roce` (Spark pairs) |
| `TENSORFOLD_GLM_MAX_IMAGES` | 128 | images per request (upstream default 50) |
| `TF_GLM_CACHE_GIB` | 0 | kept-prompt pool GiB; raise (4-12.5) to enable cross-request prefix reuse at the cost of window |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | 4 | host-memory floor for the window fit; raise if other work shares the box |

## Memory geometry (why 4bpw + 1M fits here)

The window fit is decided at startup from per-rank budgets. Measured ladder on
96 GB cards (rank 0, vision on, DFlash2):

| TF_GLM_CACHE_GIB | TENSORFOLD_MEMORY_RESERVE_GIB | largest window |
|---|---|---|
| 12.5 (Spark default) | 14.5 | 490,707 |
| 0 | 8 | 728,883 |
| 0 | 4 | **1,048,576 (full)** |

Consequences of `CACHE_GIB=0`, stated plainly: cross-request prefix reuse is
off (`kept_prompts: 0` in `/health`) — every new conversation re-prefills its
system prompt. A single deep session can own the whole window; 4 concurrent
sessions share the same 1,048,576-token pool (~260k each). If your workload is
many short agent sessions with a shared long system prompt, raise
`TF_GLM_CACHE_GIB` and accept a smaller window.

## Patches

`patches/` is MiaAi-Lab's full TensorFold recipe patch stack (0001-0053,
carried unmodified from
[GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold))
plus one addition:

- `0053-v1-models-context.patch` — `/v1/models` returns a vLLM-compatible
  object: `created`, `root`, `max_model_len` (read from the enforcement side,
  so it tracks `--context`/`KV`/`PARALLEL`), and a minimal `permission` block.
  Upstream v0.6 emits a bare `{id, object, owned_by}` and communicates the
  window only via request-time 400s.

The build stamps the patches hash into the image label `tf.patches`;
`build-image.sh` recomputes and pins it.

## Publishing

Image on Docker Hub:
[satgeze/glm-5.3-flash-exl3-4bpw-tensorfold](https://hub.docker.com/r/satgeze/glm-5.3-flash-exl3-4bpw-tensorfold)
— `v0.6.0-bc15e54e8ed6` (pinned, 2026-10-02) and `:latest`, 24.5 GB
uncompressed. `publish-docker.sh` pushes refreshed builds (version-hash + latest
tags, OCI labels, tf.patches survival check).

## Credits and licenses

- [GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
  — the checkpoint, quantized by MiaAi-Lab (EXL3/TR3 MCG, routed experts 4bpw).
- [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
  — the durable public mirror this recipe actually serves (byte-identical to
  the Mia-AiLab upload; the box copy came from here).
- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart — Apache-2.0.
  The engine; this recipe just aims it at x86_64 discrete GPUs.
- [MiaAI-Lab's Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  — the patch stack, build pipeline, and the reference two-Spark launcher this
  adapts. Their `rsync -a -L` worker-copy fix and `TENSORFOLD_GLM_MAX_IMAGES`
  patch are included here via their own patches.
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  — draft model, **CC BY-NC-ND 4.0: non-commercial use only**.
- Community prior art on this hardware class: Cardillo's 2x RTX PRO 6000 vLLM
  recipe and the verdictai SM120 images.

## Repository layout

```
patches/            54 unified diffs (upstream stack + v1-models-context)
charts/             make_charts.py + the 5 SVGs embedded above
Dockerfile          x86_64 image: NVIDA PyTorch base + TensorFold v0.6.0 + patches
build-image.sh      hash-stamped image build
download-dflash2.sh pinned DFlash2 fetch into ./hf
serve/serve-tf.sh   launcher: one container, both ranks, per-rank CUDA_VISIBLE_DEVICES
serve/start-ranks.sh  rank processes (rank1 bg, rank0 fg)
stop.sh / status.sh lifecycle + one-glance state
publish-docker.sh   Docker Hub push (version-hash + latest)
.env.example        every knob
docs/MEMORY-GEOMETRY.md   the fitting ladder and its trade-offs
```
