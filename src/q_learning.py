import numpy as np

from src.config import MAX_INVENTORY, MAX_DAYS


class QLearningAgent:
    """
    Tabular Q-Learning Agent
    """

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
    ):

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-table dimensions:
        # Inventory x Days x Actions

        self.q_table = np.zeros(
            (
                MAX_INVENTORY + 1,
                MAX_DAYS + 1,
                5,
            )
        )

    def choose_action(self, state):

        inventory, days_left = state

        if np.random.rand() < self.epsilon:

            return np.random.randint(5)

        return np.argmax(
            self.q_table[
                inventory,
                days_left
            ]
        )

    def update_q_table(
        self,
        state,
        action,
        reward,
        next_state,
    ):

        inventory, days_left = state

        next_inventory, next_days = next_state

        current_q = self.q_table[
            inventory,
            days_left,
            action
        ]

        max_future_q = np.max(
            self.q_table[
                next_inventory,
                next_days
            ]
        )

        new_q = current_q + self.lr * (
            reward
            + self.gamma * max_future_q
            - current_q
        )

        self.q_table[
            inventory,
            days_left,
            action
        ] = new_q

    def decay_epsilon(
        self,
        decay_rate=0.995,
        min_epsilon=0.01,
    ):

        self.epsilon = max(
            min_epsilon,
            self.epsilon * decay_rate
        )