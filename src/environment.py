import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DynamicPricingEnv(gym.Env):
    """
    Custom Gymnasium Environment
    for Dynamic Pricing.
    """

    def __init__(self):

        super().__init__()

        self.max_inventory = 50
        self.max_days = 30
        self.base_price = 100

        # State:
        # [remaining inventory, remaining days]

        self.observation_space = spaces.MultiDiscrete(
            [self.max_inventory + 1, self.max_days + 1]
        )

        # Five pricing actions

        self.action_space = spaces.Discrete(5)

        self.state = None

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.state = np.array(
            [self.max_inventory, self.max_days],
            dtype=np.int32,
        )

        return self.state, {}

    def step(self, action):

        raise NotImplementedError(
            "Step function will be implemented later."
        )

    def render(self):

        print(f"Inventory: {self.state[0]}")
        print(f"Days Left: {self.state[1]}")