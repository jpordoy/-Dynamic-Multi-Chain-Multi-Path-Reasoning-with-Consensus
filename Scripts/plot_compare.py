# plot_cost_vs_accuracy.py
import matplotlib.pyplot as plt

# Data
systems = [
    "Claude 3 Opus",
    "GPT-4",
    "This System\n(Haiku 3.5 + Multi-Path)",
    "Claude 3.5 Haiku\n(single-path)"
]

# X = estimated cost per 1,000 questions (in USD, take midpoint)
cost = [
    20,     # Claude 3 Opus ~$15–25
    24,     # GPT-4 ~$18–30
    3,      # This System ~$2–4
    0.8     # Claude 3.5 Haiku ~$0.6–1
]

# Y = accuracy in %
accuracy = [
    95.0,
    92.0,
    94.6,
    89.0    # approximate 88–90%
]

# Colors and markers
colors = ["blue", "green", "orange", "red"]
markers = ["o", "o", "*", "o"]

plt.figure(figsize=(10, 6))

# Scatter points
for i in range(len(systems)):
    plt.scatter(cost[i], accuracy[i], color=colors[i], s=200, marker=markers[i], label=systems[i])

# Labels and axes
plt.xlabel("Estimated Cost per 1,000 Questions (USD)")
plt.ylabel("Accuracy (%)")
plt.title("Cost vs. Accuracy Comparison")

# Grid
plt.grid(True, linestyle="--", alpha=0.6)

# Annotate each point
for i in range(len(systems)):
    plt.text(cost[i]+0.3, accuracy[i], systems[i], fontsize=9, verticalalignment='center')

# Legend
plt.legend(title="Systems", loc="lower right", frameon=True)

plt.tight_layout()

# Save figure
plt.savefig("Images/gsm8k_cost_vs_accuracy.png", dpi=200)
plt.close()

print("Plot saved as gsm8k_cost_vs_accuracy.png")
