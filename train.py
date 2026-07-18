from src.environment import DynamicPricingEnv
from src.q_learning import QLearningAgent
import numpy as np
import os

EPISODES = 500

env = DynamicPricingEnv()

agent = QLearningAgent()

episode_rewards = []

for episode in range(EPISODES):

    state, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done, truncated, info = env.step(action)

        agent.update_q_table(
            state,
            action,
            reward,
            next_state,
        )

        state = next_state

        total_reward += reward

    agent.decay_epsilon()

    episode_rewards.append(total_reward)

    if (episode + 1) % 50 == 0:

        print(
            f"Episode {episode+1}/{EPISODES}"
            f" | Revenue = {total_reward}"
            f" | Epsilon = {agent.epsilon:.3f}"
        )

print("\nTraining Complete!")

print("Final Epsilon:", agent.epsilon)

os.makedirs("models", exist_ok=True)

np.save(
    "models/q_table.npy",
    agent.q_table
)

print("Q-table saved successfully.")