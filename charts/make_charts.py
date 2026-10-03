#!/usr/bin/env python3
"""Charts for the TensorFold 4bpw dual-RTX-Pro-6000 recipe.

All numbers are measured on the same box (2x RTX PRO 6000 Blackwell 96 GB):
- TensorFold 4bpw: this recipe, 2026-10-02 (see README Results).
- vLLM 3.5bpw mixed: GLM-5.3-Flash-EXL3-3.5bpw-Mixed-SM120-TP2 README
  (1m-multi profile unless noted).
- vLLM K4 4bpw: verdictai v84 release validation (98k context).

Output: SVG files next to this script. matplotlib only.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")
OUT = os.path.dirname(os.path.abspath(__file__))

C_TF = "#2ecc71"      # TensorFold 4bpw (this recipe)
C_TF35 = "#e67e22"    # TensorFold 3.5bpw mixed (QUANT=3.5bpw arm)
C_VLLM = "#3498db"    # vLLM 3.5bpw mixed
C_K4 = "#9b59b6"      # vLLM K4 4bpw (v84)
C_REF = "#f1c40f"     # reference (Spark / other)
C_WARN = "#e74c3c"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------
# 1. Max served context per stack on this box
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.5))
stacks = [
    ("vLLM K4 4bpw\n(v84 validation)", 98304, C_K4),
    ("vLLM 3.5bpw\n(vision profile)", 700000, C_VLLM),
    ("vLLM 3.5bpw\n(1m-multi profile)", 1000000, C_VLLM),
    ("TensorFold 3.5bpw\n(QUANT=3.5bpw)", 1048560, C_TF35),
    ("TensorFold 4bpw\n(this recipe)", 1048576, C_TF),
]
labels = [s[0] for s in stacks]
vals = [s[1] for s in stacks]
cols = [s[2] for s in stacks]
x = np.arange(len(stacks))
bars = ax.bar(x, vals, 0.6, color=cols)
for xi, v in zip(x, vals):
    ax.text(xi, v + 12000, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Max served context (tokens)", fontsize=10)
ax.set_ylim(0, 1180000)
ax.set_title("Max served context on this box (2x RTX PRO 6000, 192 GB)\n"
             "same checkpoint family, different engines and quants",
             fontsize=12, fontweight="bold", pad=12)
ax.axhline(1048576, color=C_TF, lw=0.7, ls=":")
ax.text(4.35, 1055000, "model native window", fontsize=8, color="#aaaaaa")
save(fig, "context-by-stack.svg")

# ---------------------------------------------------------------
# 2. Single-stream decode, thinking on
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
names = ["TensorFold 4bpw\n(this recipe)", "vLLM 3.5bpw\n(1m-multi profile)"]
vals = [63.2, 143.0]
notes = ["thinking-on, 2k prompt, 200 deltas", "thinking-on, warm, forced-gen protocol"]
bars = ax.bar(np.arange(2), vals, 0.5, color=[C_TF, C_VLLM])
for xi, (v, n) in enumerate(zip(vals, notes)):
    ax.text(xi, v + 3, f"{v} tok/s", ha="center", fontsize=12, fontweight="bold")
    ax.text(xi, v / 2, n, ha="center", fontsize=9, color="#111111")
ax.set_xticks(np.arange(2))
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Decode tok/s (engine deltas)", fontsize=10)
ax.set_ylim(0, 165)
ax.set_title("Single-stream decode, thinking on\n"
             "TensorFold counts engine deltas; vLLM number from the 1m-multi README (143 tok/s thinking-on)",
             fontsize=12, fontweight="bold", pad=12)
save(fig, "decode-single.svg")

# ---------------------------------------------------------------
# 3. Aggregate decode at 4 concurrent streams
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))
names = ["TensorFold 4bpw\nprose @4", "TensorFold 4bpw\nJSON @4", "vLLM 3.5bpw\n(1m-multi) @4"]
vals = [325.1, 375.2, 141.2]
cols = [C_TF, C_TF, C_VLLM]
bars = ax.bar(np.arange(3), vals, 0.55, color=cols)
for xi, v in enumerate(vals):
    ax.text(xi, v + 6, f"{v} tok/s", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(np.arange(3))
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Aggregate decode tok/s", fontsize=10)
ax.set_ylim(0, 430)
ax.set_title("Aggregate decode, 4 concurrent streams\n"
             "TensorFold: completion tokens / wall; vLLM: 1m-multi README aggregate @4 (141.2)",
             fontsize=12, fontweight="bold", pad=12)
save(fig, "decode-concurrency.svg")

# ---------------------------------------------------------------
# 4. Prefill throughput
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.5))
names = ["vLLM 3.5bpw\n@500K", "vLLM 3.5bpw\n@750K", "vLLM 3.5bpw\n@950K",
         "TensorFold 4bpw\n@128k", "TensorFold 4bpw\n@1.008M"]
vals = [2836.7, 2840.6, 2793.2, 3470.0, 2379.0]
cols = [C_VLLM] * 3 + [C_TF] * 2
bars = ax.bar(np.arange(5), vals, 0.6, color=cols)
for xi, v in enumerate(vals):
    ax.text(xi, v + 40, f"{v:,.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(np.arange(5))
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Prefill tok/s", fontsize=10)
ax.set_ylim(0, 3900)
ax.set_title("Prefill throughput\n"
             "vLLM: stream-TTFT at depth (README); TensorFold: TTFT-derived @128k, needle wall-derived @1.008M (methods differ)",
             fontsize=12, fontweight="bold", pad=12)
save(fig, "prefill.svg")

# ---------------------------------------------------------------
# 5. GSM8K (250-problem test slice where noted)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))
names = ["vLLM 3.5bpw\n(this box)", "TensorFold 4bpw\n(this box, greedy)",
         "TensorFold 3.5bpw\n(this box, greedy)", "TensorFold\n(Spark, Mia 250)"]
vals = [96.89, 97.2, 98.4, 98.0]
cols = [C_VLLM, C_TF, C_TF35, C_REF]
bars = ax.bar(np.arange(4), vals, 0.5, color=cols)
for xi, v in enumerate(vals):
    ax.text(xi, v + 0.15, f"{v}%", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(np.arange(4))
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("GSM8K accuracy %", fontsize=10)
ax.set_ylim(90, 100)
ax.set_title("GSM8K accuracy\nTensorFold box: 250-problem test slice, greedy; vLLM 3.5bpw: recipe README; Spark: Mia's recipe (250)",
             fontsize=12, fontweight="bold", pad=12)
save(fig, "gsm8k.svg")

print("all charts written")
