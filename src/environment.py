import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.demand import DemandModel
from src.config import (
    MAX_INVENTORY,
    MAX_DAYS,
    BASE_PRICE,
    PRICE_LEVELS
)

class DynamicPricingEnv(gym.Env):
    """
    Custom Gymnasium Environment
    for Dynamic Pricing.
    """

    def __init__(self):

        super().__init__()

        self.max_inventory = MAX_INVENTORY
        self.max_days = MAX_DAYS
        self.base_price = BASE_PRICE
        self.price_levels = PRICE_LEVELS

        # State:
        # [remaining inventory, remaining days]

        self.observation_space = spaces.MultiDiscrete(
            [self.max_inventory + 1, self.max_days + 1]
        )

        # Five pricing actions

        self.action_space = spaces.Discrete(5)

        self.state = None

        self.demand_model = DemandModel()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.state = np.array(
            [self.max_inventory, self.max_days],
            dtype=np.int32,
        )

        return self.state, {}

    def step(self, action):

        inventory, days_left = self.state

        done = False

        reward = 0

        if inventory > 0:

            bought = self.demand_model.customer_buys(action, days_left)

            if bought:
                inventory -= 1
                reward = self.price_levels[action]

        days_left -= 1

        if inventory == 0 or days_left == 0:
            done = True

        self.state = np.array(
            [inventory, days_left],
            dtype=np.int32
        )

        return self.state, reward, done, False, {}

    def render(self):

        print(f"Inventory: {self.state[0]}")
        print(f"Days Left: {self.state[1]}")