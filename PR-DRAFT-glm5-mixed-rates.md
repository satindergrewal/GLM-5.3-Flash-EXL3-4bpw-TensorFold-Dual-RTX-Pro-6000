# DRAFT PR — for Satinder's review only. Nothing pushed anywhere.

Target: ashhart/TensorFold, base `main` (v0.6.5)
Branch (to be cut from upstream/main when approved): `glm5-mixed-rates`
Estimated size: ~600 lines + tests, 5 commits

---

## Title

glm5_next: mixed k3/k4 per-tensor EXL3 rates (per-expert-width routed experts)

## Summary

Adds support for EXL3 checkpoints that pick a bit rate per tensor instead of one
rate for the whole checkpoint (for example Mia-Ai-Lab's GLM-5.3-Flash 3.5bpw
mixed encode on Hugging Face: half the expert tensors at k3, half at k4, with
`"bits": "mixed_k34_per_tensor"` in `quantization_config`).

Today such a checkpoint is refused by name: `Config.bits_of` raises for any
non-integer bits marker. This PR accepts the mixed marker for the glm5_next
family, reads each expert tensor's width from its own trellis shape, and routes
mixed-width layers through the per-expert-width `Exl3RoutedExperts` ABI that
`tensorfold.cuda.exl3.experts` already provides (`prepare`).

## Motivation

GLM-5.3-Flash is a 320B MoE that does not fit a 192 GB (2x96 GB) box at 4 bpw
with a usable context under vLLM. The mixed 3.5bpw encode does, with measured
quality at parity (GSM8K 250-problem slice: 98.4% mixed vs 97.2% stock 4bpw vs
96.89% vLLM serving the same mixed artifact, all on the same 2x RTX PRO 6000
box; five-run KLD gate vs teacher at encode time: mean 0.0246, bar 0.06).
Serving that checkpoint under TensorFold needs exactly this patch.

## What the checkpoint looks like

- `quantization_config.bits = "mixed_k34_per_tensor"`, `mixed: true`,
  `codebook: mcg`, `allowed_bits: [3, 4]`
- expert trellises stored int16-native, `[K/16, N/16, 16*K/2]`:
  48-wide (k3) or 64-wide (k4) per tensor; 4,880 experts pure k4, 4,823 pure
  k3, 2,681 split-projection (different rates per projection)
- suh/svh fp16 per tensor; a `mcg` marker tensor per weight

## Implementation

Five pieces, all inside the glm5_next family plus one generic-module touch:

1. `Config` parse (weights.py): a non-integer bits marker falls back to 4 for
   geometry only; every tensor's real width comes from its own trellis shape.
   `bits_of` keeps refusing unknown non-integer markers; the known
   `mixed_k34_per_tensor` marker is accepted for `quant_method == "exl3"` with
   `codebook == "mcg"`.
2. Family gate (`__init__.py`): accepts the mixed mcg variant
   (`allowed_bits [3, 4]`).
3. `exl3_mm.words`: accepts 3-bit `int16 [..., 48]` alongside 4-bit
   `[..., 64]`.
4. `moe_exl3` (weights.py): reads all 288 experts' trellis widths; uniform
   layers take `prepare_stacked` (unchanged behavior); mixed layers take
   `prepare` with per-expert (trellis, suh, svh) triples - the existing
   per-expert-width ABI, no new kernels.
5. `moe_block` (forward.py): when the layer's experts are per-expert-width,
   the routed pass slices the pick to the contiguous top-k columns and runs in
   scratch-sized row slices.

Piece 5 needs a paragraph: the group kernel indexes the pick tensor with flat
`r * slots + s` arithmetic against the scratch's slot count. The shared expert
rides the pick tensor as an extra column, so passing the full
`[R, top_k + 1]` pick against a top_k-slot scratch shifts every row's expert
window by one position from row 1 on - each token silently loses its last
expert and inherits the previous row's picks. Slicing to `[R, :top_k]`
contiguous makes the strides agree. (Upstream's own route avoids this by
building the scratch with `top_k + 1` slots and letting the kernels compute
the skipped shared slot; this PR's slices instead, which also skips ~1/9 of
the routed-pass work per token.)

## Measured (2x RTX PRO 6000 Blackwell 96 GB, TensorFold 0.6.5 + this change)

- Boot: 1,048,560-token context served; all 45 MoE layers report mixed widths
  [48, 64] and load through the per-expert-width path.
- GSM8K, 250-problem test slice, greedy: **98.4% (246/250)** - above the same
  artifact served by vLLM (96.89%) and the stock 4bpw checkpoint on the same
  engine (97.2%).
- Long-context: needle retrieved exactly at 90% depth of a ~912k-token prompt.
- Vision: 3-image color-ID pass (the checkpoint ships the image-capable chat
  template; the vision tower is bf16 and quantization-independent).
- DFlash2 drafting: 74.9% token acceptance across a benchmark session.

## Tests

- `bits_of` / Config parse: the mixed marker is accepted for glm5_next exl3+mcg
  and still refused for other families and unknown markers.
- `prepare` with mixed widths: k2 array carries per-expert half-bits; a
  dequant round-trip matches the python reference for k3 and k4 tensors.
- The pick-slice contract: a [R, top_k+1] pick and a top_k-slot scratch agree
  with a python routing reference, per (row, slot), for R in {1, 8, 1024, 2048}.
- Existing suite passes unchanged.

## Commits (planned series on the branch)

1. glm5_next: accept the mixed k3/k4 per-tensor exl3 marker (config + gate)
2. glm5_next: accept 3-bit trellis widths in exl3_mm.words
3. cuda/exl3: per-expert-width routed experts through prepare (no code change
   expected here; assert the ABI this PR relies on)
4. glm5_next: per-expert-width moe_exl3 loader (uniform keeps prepare_stacked)
5. glm5_next: routed pass picks the contiguous top-k columns, sliced to the
   scratch's rows
6. tests

## Notes

- No changes outside glm5_next except asserting the existing
  `cuda/exl3/experts` ABI; the update-check module, recipe defaults and
  anything deployment-specific are not part of this PR.
- The mixed encode this was built against is public
  (Mia-Ai-Lab GLM-5.3-Flash EXL3 3.5bpw mixed), so the reviewer can reproduce
  the numbers.
