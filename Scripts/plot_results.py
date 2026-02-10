"""
plot_results.py
===============
Generates a bar chart comparing GSM8K accuracy across models.

Usage:
    pip install matplotlib
    python plot_results.py

Saves:  gsm8k_comparison.png  (in current directory)

References:
    [1] OpenAI. GPT-4 Technical Report. arXiv:2303.08774 (2023).
    [2] Anthropic. Claude 3 Model Card (March 2024).
    [3] This work: 946/1000 questions correct.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────

models = [
    "GPT-3.5 Turbo\n(5-shot) [1]",
    "Claude 3 Haiku\n(0-shot, est.) [2]",
    "GPT-4\n(0-shot CoT) [1]",
    "This System\n(Haiku 3.5 + Multi-Path) ★",
    "Claude 3 Opus\n(0-shot CoT) [2]",
]

accuracies = [57.1, 88.9, 92.0, 94.6, 95.0]

# Colour: highlight this system in a distinct colour
colors = ["#9e9e9e", "#9e9e9e", "#9e9e9e", "#00C853", "#9e9e9e"]
edgecolors = ["#616161", "#616161", "#616161", "#007A33", "#616161"]

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#0d1117")   # GitHub dark background
ax.set_facecolor("#161b22")

y_pos = np.arange(len(models))
bars = ax.barh(
    y_pos,
    accuracies,
    color=colors,
    edgecolor=edgecolors,
    height=0.55,
    linewidth=1.2,
)

# Value labels
for bar, acc in zip(bars, accuracies):
    ax.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{acc:.1f}%",
        va="center",
        ha="left",
        fontsize=10,
        color="#e6edf3",
        fontweight="bold" if acc == 94.6 else "normal",
    )

# Axes
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=9, color="#e6edf3")
ax.set_xlabel("GSM8K Accuracy (%)", fontsize=10, color="#8b949e", labelpad=10)
ax.set_xlim(50, 100)
ax.xaxis.set_tick_params(colors="#8b949e")
ax.tick_params(axis="x", colors="#8b949e")

# Grid
ax.xaxis.grid(True, color="#30363d", linewidth=0.8, linestyle="--")
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

# Title
ax.set_title(
    "GSM8K Benchmark Accuracy — Model Comparison\n"
    "★ Multi-path inference-time scaling (not directly comparable to 0-shot baselines)",
    fontsize=11,
    color="#e6edf3",
    pad=14,
    loc="left",
)

# Legend
star_patch = mpatches.Patch(color="#00C853", label="This work (Claude 3.5 Haiku + Multi-Path)")
grey_patch = mpatches.Patch(color="#9e9e9e", label="Published baselines (0-shot CoT unless noted)")
ax.legend(
    handles=[star_patch, grey_patch],
    loc="lower right",
    fontsize=8,
    facecolor="#161b22",
    edgecolor="#30363d",
    labelcolor="#e6edf3",
)

# References footnote
fig.text(
    0.01, 0.01,
    "[1] OpenAI, GPT-4 Technical Report, arXiv:2303.08774 (2023)   "
    "[2] Anthropic, Claude 3 Model Card (2024)",
    fontsize=7,
    color="#8b949e",
    ha="left",
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("gsm8k_comparison.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: gsm8k_comparison.png")
plt.show()
