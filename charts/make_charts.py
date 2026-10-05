#!/usr/bin/env python3
"""Charts for the TensorFold dual-arm GLM-5.3-Flash recipe (4bpw stock + 3.5bpw mixed).

All numbers are measured on the same box (2x RTX PRO 6000 Blackwell 96 GB):
- TensorFold 4bpw: this recipe, 2026-10-02 (see README Results).
- TensorFold 3.5bpw mixed (QUANT=3.5bpw): 2026-10-04, post pick-stride fix.
- vLLM 3.5bpw mixed: GLM-5.3-Flash-EXL3-3.5bpw-Mixed-SM120-TP2 README
  (1m-multi profile unless noted).
- vLLM K4 4bpw: v84 release validation (98k context).

Output: SVG files next to this script (and a PNG copy in /tmp for review).
matplotlib only.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.facecolor": "#111111",
    "axes.facecolor": "#111111",
    "savefig.facecolor": "#111111",
})
plt.style.use("dark_background")
OUT = os.path.dirname(os.path.abspath(__file__))
TMP = "/tmp"

C_TF = "#2ecc71"      # TensorFold 4bpw (this recipe)
C_TF35 = "#e67e22"    # TensorFold 3.5bpw mixed (QUANT=3.5bpw arm)
C_VLLM = "#3498db"    # vLLM 3.5bpw mixed
C_K4 = "#9b59b6"      # vLLM K4 4bpw (v84)
C_REF = "#f1c40f"     # reference (Spark / other)
GRID = "#333333"
SUB = "#bbbbbb"


def save(fig, name):
    for path in (os.path.join(OUT, name), os.path.join(TMP, name.replace(".svg", ".png"))):
        fig.savefig(path, format="svg" if path.endswith(".svg") else "png",
                    dpi=110 if path.endswith(".png") else None,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def caption(fig, text):
    fig.text(0.5, 0.885, text, ha="center", fontsize=12, color=SUB)


def style(ax, title, ylab, sub=None):
    ax.set_title(title, fontsize=17, fontweight="bold", pad=28)
    if sub:
        caption(ax.figure, sub)
    ax.set_ylabel(ylab)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def label_bars(ax, bars, vals, fmt="{:,.0f}", dy=0.02):
    top = ax.get_ylim()[1]
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + top * dy, fmt.format(v),
                ha="center", fontsize=15, fontweight="bold")


# ---------------------------------------------------------------
# 1. Max served context per stack on this box
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6.2))
stacks = [
    ("vLLM K4 4bpw\n(v84, 98k)", 98304, C_K4),
    ("vLLM 3.5bpw\n(vision, 700k)", 700000, C_VLLM),
    ("vLLM 3.5bpw\n(1m-multi)", 1000000, C_VLLM),
    ("TensorFold 3.5bpw\n(mixed)", 1048560, C_TF35),
    ("TensorFold 4bpw\n(stock)", 1048576, C_TF),
]
x = np.arange(len(stacks))
bars = ax.bar(x, [s[1] for s in stacks], 0.62, color=[s[2] for s in stacks])
for b, (_, v, _) in zip(bars, stacks):
    ax.text(b.get_x() + b.get_width() / 2, v + 22000, f"{v:,}", ha="center", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in stacks])
ax.set_ylim(0, 1220000)
ax.set_yticks(np.arange(0, 1200001, 300000))
ax.set_yticklabels(["0", "300k", "600k", "900k", "1.2M"])
style(ax, "Max served context on this box", "Tokens (native window = 1,048,576)",
      "2x RTX PRO 6000 (192 GB) - same checkpoint family, different engines and quants")
save(fig, "context-by-stack.svg")

# ---------------------------------------------------------------
# 2. Single-stream decode, thinking on
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.2))
names = ["TensorFold 4bpw", "TensorFold 3.5bpw\n(mixed)", "vLLM 3.5bpw\n(1m-multi)"]
vals = [63.2, 148.0, 143.0]
bars = ax.bar(np.arange(3), vals, 0.5, color=[C_TF, C_TF35, C_VLLM])
for i, (b, v) in enumerate(zip(bars, vals)):
    ax.text(b.get_x() + b.get_width() / 2, v + 3.5, ("130-158 tok/s" if i == 1 else f"{v} tok/s"), ha="center", fontsize=15, fontweight="bold")
ax.set_xticks(np.arange(3))
ax.set_xticklabels(names)
ax.set_ylim(0, 200)
ax.set_yticks(np.arange(0, 201, 50))
style(ax, "Single-stream decode, thinking on", "tok/s (engine deltas)",
      "release engine config; the paired-kernel build trades some single-stream decode (universal path: 163-175) for prefill")
save(fig, "decode-single.svg")

# ---------------------------------------------------------------
# 3. Aggregate decode at 4 concurrent streams
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.5, 6.2))
names = ["TensorFold 4bpw\nprose", "TensorFold 4bpw\nJSON", "TensorFold 3.5bpw\nprose", "vLLM 3.5bpw\n(1m-multi)"]
vals = [325.1, 375.2, 339.0, 141.2]
labels = ["325.1 tok/s", "375.2 tok/s", "334-339 tok/s", "141.2 tok/s"]
bars = ax.bar(np.arange(4), vals, 0.55, color=[C_TF, C_TF, C_TF35, C_VLLM])
for b, v, lab in zip(bars, vals, labels):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, lab, ha="center", fontsize=15, fontweight="bold")
ax.set_xticks(np.arange(4))
ax.set_xticklabels(names)
ax.set_ylim(0, 430)
ax.set_yticks(np.arange(0, 401, 100))
style(ax, "Aggregate decode, 4 concurrent streams", "tok/s (completion tokens / wall)",
      "3.5bpw paired tuned kernels (fork v0.6.5-pair)")
save(fig, "decode-concurrency.svg")

# ---------------------------------------------------------------
# 4. Prefill throughput
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.5, 6.2))
names = ["vLLM 3.5bpw\n@500K", "vLLM 3.5bpw\n@750K", "vLLM 3.5bpw\n@950K",
         "TensorFold 4bpw\n@128k", "TensorFold 4bpw\n@1.008M", "TensorFold 3.5bpw\n@166k *"]
vals = [2836.7, 2840.6, 2793.2, 3470.0, 2379.0, 3020.0]
bars = ax.bar(np.arange(6), vals, 0.6, color=[C_VLLM] * 3 + [C_TF] * 2 + [C_TF35])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 55, f"{v:,.0f}", ha="center", fontsize=14, fontweight="bold")
ax.set_xticks(np.arange(6))
ax.set_xticklabels(names)
ax.set_ylim(0, 3950)
ax.set_yticks(np.arange(0, 3501, 1000))
style(ax, "Prefill throughput", "tok/s",
      "TTFT-derived; 3.5bpw @166k: lanes + NCCL match + paired tuned kernels (width-64 experts on the Y^T prompt kernels), 54.4-55.2 s cold TTFT (peer DMA silently drops data on this box's mixed GPU pair, so all engines ride host transports)")
save(fig, "prefill.svg")

# ---------------------------------------------------------------
# 5. GSM8K (250-problem test slice where noted)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.2))
names = ["vLLM 3.5bpw\n(this box)", "TensorFold 4bpw\n(this box)", "TensorFold 3.5bpw\n(this box)", "TensorFold\n(Spark, Mia)"]
vals = [96.89, 97.2, 98.4, 98.0]
bars = ax.bar(np.arange(4), vals, 0.5, color=[C_VLLM, C_TF, C_TF35, C_REF])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v}%", ha="center", fontsize=15, fontweight="bold")
ax.set_xticks(np.arange(4))
ax.set_xticklabels(names)
ax.set_ylim(90, 100)
ax.set_yticks(np.arange(90, 101, 2))
style(ax, "GSM8K accuracy, 250-problem test slices, greedy", "Accuracy %",
      "all runs on 2x RTX PRO 6000 on this box except the Spark reference (Mia's recipe)")
save(fig, "gsm8k.svg")

print("all charts written")
