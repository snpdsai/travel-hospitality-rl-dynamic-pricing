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