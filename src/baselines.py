import numpy as np

from src.environment import DynamicPricingEnv

def fixed_price_strategy(price_action=2):
    """
    Always selects the same pricing action.

    Default action = 2
    Corresponds to price = 120.
    """

    env = DynamicPricingEnv()

    state, info = env.reset()

    done = False

    total_reward = 0

    while not done:

        state, reward, terminated, truncated, info = env.step(price_action)

        total_reward += reward

        done = terminated or truncated

    return total_reward

def time_discount_strategy():
    """
    High prices initially.
    Gradually reduce prices as departure approaches.
    """

    env = DynamicPricingEnv()

    state, info = env.reset()

    done = False

    total_reward = 0

    while not done:

        days = env.days_left

        if days > 12:
            action = 4          # $160
        elif days > 9:
            action = 3          # $140
        elif days > 6:
            action = 2          # $120
        elif days > 3:
            action = 1          # $100
        else:
            action = 0          # $80

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        done = terminated or truncated

    return total_reward