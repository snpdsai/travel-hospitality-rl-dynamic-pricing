import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

rewards = np.load("models/training_rewards.npy")

plt.figure(figsize=(10,5))
plt.plot(rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Revenue")
plt.title("Q-Learning Training Rewards")
plt.legend()

plt.tight_layout()

plt.savefig("outputs/figures/qlearning_training_curve.png")

plt.show()