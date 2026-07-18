from src.environment import DynamicPricingEnv
import numpy as np
from src.config import DEFAULT_EPISODES
from src.q_learning import QLearningAgent

def run_episode(agent):

    env = DynamicPricingEnv()

    state, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        action = agent.choose_action(state)

        state, reward, done, truncated, info = env.step(action)

        total_reward += reward

    return total_reward


def evaluate_agent(agent, episodes=DEFAULT_EPISODES):

    rewards = []

    for _ in range(episodes):
        rewards.append(run_episode(agent))

    return {
        "Average Revenue": np.mean(rewards),
        "Std Revenue": np.std(rewards),
        "Min Revenue": np.min(rewards),
        "Max Revenue": np.max(rewards),
    }

def evaluate_qlearning(agent, episodes=100):

    env = DynamicPricingEnv()

    rewards = []

    for _ in range(episodes):

        state, _ = env.reset()

        done = False

        total_reward = 0

        while not done:

            inventory, days_left = state

            action = agent.q_table[
                inventory,
                days_left
            ].argmax()

            state, reward, done, truncated, info = env.step(action)

            total_reward += reward

        rewards.append(total_reward)

    return {
        "Average Revenue": np.mean(rewards),
        "Std Revenue": np.std(rewards),
        "Min Revenue": np.min(rewards),
        "Max Revenue": np.max(rewards),
    }