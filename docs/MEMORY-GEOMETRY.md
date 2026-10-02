# Memory geometry on 2x RTX PRO 6000 (96 GB each)

TensorFold fits the model, the window, and any extra prompt-cache pool into a
per-rank budget it computes at startup. On the DGX Spark (128 GB unified) the
stock recipe defaults leave room for everything; on 96 GB discrete cards the
defaults cost you more than half the window. Measured ladder (rank 0, vision
on, DFlash2 drafting, TP2):

| TF_GLM_CACHE_GIB | TENSORFOLD_MEMORY_RESERVE_GIB | largest fitting window |
|---|---|---|
| 12.5 (Spark default) | 14.5 | 490,707 |
| 0 | 8 | 728,883 |
| 0 | 4 | **1,048,576** |

Final geometry at the last row: rank 0 estimate 88.09 GiB within 90.08 GiB;
rank 1 86.29 GiB. Enforced per-request limit: 1,048,560 (window minus a small
reply reserve).

## What each knob buys and costs

- `TF_GLM_CACHE_GIB` — the kept-prompt pool *beyond* the window: resumed
  prompt states across conversations (identical system prompt re-prefill goes
  from ~34 s to under 0.07 s at 64k). At 0, `kept_prompts` stays 0 and every
  new conversation prefills fresh. Within a single request, incremental
  decode and chunked prefill are unaffected.
- `TENSORFOLD_MEMORY_RESERVE_GIB` — headroom the fitter keeps free. On a
  dedicated box 4 is enough; raise it if anything else shares the GPUs or
  host memory.
- `--context N` — pin the window explicitly (e.g. 728883) instead of
  auto-fitting; do this when you want a *deterministic* refusal boundary for
  clients.
- `--parallel` — decode streams sharing the window's pool. 4 streams ≈ 4×260k
  usable each; one deep session can own the entire 1,048,576.
- `VISION=0` — dropping the vision tower (rank 0) frees ~1.8 GiB of weights
  plus prefill transients if you need a little more window and don't need
  images.

## Failure modes seen while fitting

- Over-budget window: clean startup refusal with the exact largest fitting
  number (`--context 0` then takes that number automatically).
- Memory guard: the engine kills itself when free memory crosses its floor
  rather than wedging the host (this behavior is what makes thin headroom
  survivable).
