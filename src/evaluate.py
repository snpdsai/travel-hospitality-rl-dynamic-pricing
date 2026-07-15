from src.environment import DynamicPricingEnv
import numpy as np
from src.config import DEFAULT_EPISODES

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