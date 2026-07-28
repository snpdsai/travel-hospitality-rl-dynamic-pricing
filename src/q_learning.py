"""
Tabular Q-Learning implementation for
Dynamic Pricing.
"""

from __future__ import annotations

import numpy as np

from src.environment import DynamicPricingEnv
from src.config import (
    QL_ALPHA,
    QL_GAMMA,
    QL_EPSILON_START,
    QL_EPSILON_DECAY,
    QL_EPSILON_MIN,
)


class QLearningAgent:
    """
    Tabular Q-Learning Agent.
    """

    def __init__(self):

        self.env = DynamicPricingEnv()

        self.alpha = QL_ALPHA
        self.gamma = QL_GAMMA

        self.epsilon = QL_EPSILON_START
        self.epsilon_decay = QL_EPSILON_DECAY
        self.epsilon_min = QL_EPSILON_MIN

        inventory_states = self.env.initial_inventory + 1
        day_states = self.env.booking_horizon + 1
        actions = self.env.action_space.n

        self.q_table = np.zeros(
            (
                inventory_states,
                day_states,
                actions,
            ),
            dtype=np.float32,
        )

    def choose_action(self, state):
        """
        Choose an action using an epsilon-greedy policy.
        """

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        inventory, days = state

        return int(np.argmax(self.q_table[inventory, days]))

    def update_q_table(
        self,
        state,
        action,
        reward,
        next_state,
    ):
        """
        Update the Q-table using the Bellman equation.
        """

        inventory, days = state
        next_inventory, next_days = next_state

        current_q = self.q_table[
            inventory,
            days,
            action,
        ]

        max_future_q = np.max(
            self.q_table[
                next_inventory,
                next_days,
            ]
        )

        updated_q = current_q + self.alpha * (
            reward
            + self.gamma * max_future_q
            - current_q
        )

        self.q_table[
            inventory,
            days,
            action,
        ] = updated_q

    def decay_epsilon(self):
        """
        Reduce exploration after each episode.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def train(self, episodes=500):
        """
        Train the Q-Learning agent.
        """

        episode_rewards = []

        for episode in range(episodes):

            state, _ = self.env.reset()

            done = False

            total_reward = 0

            while not done:

                action = self.choose_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                self.update_q_table(
                    state,
                    action,
                    reward,
                    next_state,
                )

                state = next_state

                total_reward += reward

                done = terminated or truncated

            self.decay_epsilon()

            episode_rewards.append(total_reward)

            if (episode + 1) % 50 == 0:

                print(
                    f"Episode {episode + 1}/{episodes} | "
                    f"Revenue = {total_reward} | "
                    f"Epsilon = {self.epsilon:.3f}"
                )

        return episode_rewards

    def evaluate(self, episodes=100):
        """
        Evaluate the learned policy without exploration.
        """

        revenues = []

        original_epsilon = self.epsilon

        self.epsilon = 0.0

        for _ in range(episodes):

            state, _ = self.env.reset()

            done = False

            total_reward = 0

            while not done:

                inventory, days = state

                action = int(
                    np.argmax(
                        self.q_table[
                            inventory,
                            days,
                        ]
                    )
                )

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                total_reward += reward

                state = next_state

                done = terminated or truncated

            revenues.append(total_reward)

        self.epsilon = original_epsilon

        return revenues