import os
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

agents = [
    "Fixed",
    "Time\nDiscount",
    "Q-Learning"
]

revenues = [
    1524.0,
    1490.2,
    1513.2
]

plt.figure(figsize=(6,5))
plt.bar(agents, revenues)

plt.ylabel("Average Revenue")
plt.title("Average Revenue Comparison")

plt.tight_layout()

plt.savefig("outputs/figures/revenue_comparison.png")

plt.show()