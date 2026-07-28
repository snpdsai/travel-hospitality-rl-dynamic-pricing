"""
Custom Gymnasium environment for
Dynamic Pricing using Reinforcement Learning.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import (
    INITIAL_INVENTORY,
    BOOKING_HORIZON,
    PRICE_LEVELS,
)


class DynamicPricingEnv(gym.Env):
    """
    Airline / Hotel Dynamic Pricing Environment.

    State:
        [remaining_inventory, remaining_days]

    Actions:
        0 -> $80
        1 -> $100
        2 -> $120
        3 -> $140
        4 -> $160

    Reward:
        Revenue generated from a successful booking.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.initial_inventory = INITIAL_INVENTORY
        self.booking_horizon = BOOKING_HORIZON
        self.price_levels = PRICE_LEVELS

        self.action_space = spaces.Discrete(len(self.price_levels))

        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array(
                [
                    self.initial_inventory,
                    self.booking_horizon,
                ],
                dtype=np.int32,
            ),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        """

        super().reset(seed=seed)

        self.inventory = self.initial_inventory
        self.days_left = self.booking_horizon

        state = np.array(
            [
                self.inventory,
                self.days_left,
            ],
            dtype=np.int32,
        )

        return state, {}

    def demand_probability(self, price: int) -> float:
        """
        Estimate customer purchase probability based on
        price, remaining booking time, and remaining inventory.
        """

        base_probability = {
            80: 0.90,
            100: 0.75,
            120: 0.60,
            140: 0.40,
            160: 0.25,
        }

        probability = base_probability[price]

        # --------------------------------------------------
        # Demand increases as departure approaches
        # --------------------------------------------------

        time_progress = (
            self.booking_horizon - self.days_left
        ) / self.booking_horizon

        probability += 0.45 * (time_progress ** 1.5)

        # --------------------------------------------------
        # Encourage selling remaining inventory
        # --------------------------------------------------

        inventory_ratio = self.inventory / self.initial_inventory

        if inventory_ratio > 0.70:
            probability += 0.10

        elif inventory_ratio < 0.30:
            probability -= 0.05

        probability = np.clip(
            probability,
            0.05,
            0.99,
        )

        return float(probability)

    def step(self, action):
        """
        Execute one pricing decision.
        """

        price = self.price_levels[action]

        probability = self.demand_probability(price)

        purchase = (
            np.random.random() < probability
            and self.inventory > 0
        )

        reward = 0

        if purchase:
            reward = price
            self.inventory -= 1

        self.days_left -= 1

        terminated = (
            self.days_left <= 0
            or self.inventory <= 0
        )

        # Penalize unsold inventory when the booking season ends
        if self.days_left <= 0 and self.inventory > 0:
            reward -= 20 * self.inventory

        truncated = False

        next_state = np.array(
            [
                self.inventory,
                self.days_left,
            ],
            dtype=np.int32,
        )

        info = {
            "price": price,
            "probability": probability,
            "purchase": purchase,
        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        """
        Display the current environment state.
        """

        print(
            f"Inventory: {self.inventory} | "
            f"Days Left: {self.days_left}"
        )