import random

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    STATE_SIZE,
    ACTION_SIZE,
    HIDDEN_SIZE,
    LEARNING_RATE,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY
)


class DQN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),

            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),

            nn.Linear(HIDDEN_SIZE, ACTION_SIZE)
        )

    def forward(self, x):

        return self.network(x)


class DQNAgent:

    def __init__(self):

        self.model = DQN()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

        self.loss_fn = nn.MSELoss()

        self.epsilon = EPSILON_START

    def choose_action(self, state):

        if random.random() < self.epsilon:

            return random.randint(
                0,
                ACTION_SIZE - 1
            )

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.model(state_tensor)

        return torch.argmax(
            q_values
        ).item()

    def decay_epsilon(self):

        self.epsilon = max(
            EPSILON_MIN,
            self.epsilon * EPSILON_DECAY
        )