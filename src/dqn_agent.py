import torch
import torch.nn as nn

from src.config import (
    STATE_SIZE,
    ACTION_SIZE,
    HIDDEN_SIZE
)


class DQN(nn.Module):
    """
    Deep Q-Network
    """

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                STATE_SIZE,
                HIDDEN_SIZE
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_SIZE,
                HIDDEN_SIZE
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_SIZE,
                ACTION_SIZE
            )
        )

    def forward(self, x):

        return self.network(x)