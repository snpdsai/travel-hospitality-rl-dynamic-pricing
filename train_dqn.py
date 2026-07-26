import os
import numpy as np

from src.environment import DynamicPricingEnv
from src.dqn_agent import DQNAgent
from src.replay_buffer import ReplayBuffer

from src.config import (
    DQN_EPISODES,
    START_TRAINING_AFTER,
    TARGET_UPDATE_FREQUENCY
)

os.makedirs("models", exist_ok=True)

env = DynamicPricingEnv()

agent = DQNAgent()

buffer = ReplayBuffer()

episode_rewards = []

for episode in range(DQN_EPISODES):

    state, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done, truncated, info = env.step(action)

        buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        total_reward += reward

        if len(buffer) >= START_TRAINING_AFTER:

            agent.train_step(buffer)

    agent.decay_epsilon()

    if (episode + 1) % TARGET_UPDATE_FREQUENCY == 0:

        agent.update_target_network()

    episode_rewards.append(total_reward)

    if (episode + 1) % 50 == 0:

        print(
            f"Episode {episode+1}/{DQN_EPISODES}"
            f" | Revenue = {total_reward}"
            f" | Epsilon = {agent.epsilon:.3f}"
        )

agent.save_model(
    "models/dqn_model.pth"
)

np.save(
    "models/dqn_rewards.npy",
    np.array(episode_rewards)
)

print("\nTraining Complete!")

print("Model saved.")

print("Rewards saved.")