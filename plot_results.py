import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

rewards = np.load("models/training_rewards.npy")

window = 20

moving_average = np.convolve(
    rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(10,5))

plt.plot(
    rewards,
    alpha=0.3,
    label="Episode Reward"
)

plt.plot(
    range(window-1, len(rewards)),
    moving_average,
    linewidth=2,
    label="20-Episode Moving Average"
)

plt.xlabel("Episode")
plt.ylabel("Revenue")
plt.title("Q-Learning Training Performance")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/figures/qlearning_training_curve.png"
)

plt.show()