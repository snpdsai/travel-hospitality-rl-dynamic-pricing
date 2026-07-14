from src.environment import DynamicPricingEnv


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