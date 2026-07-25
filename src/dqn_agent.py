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
    EPSILON_DECAY,
    GAMMA,
    BATCH_SIZE
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

        # Target Network
        self.target_model = DQN()

        # Initial synchronization
        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.target_model.eval()

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

    def update_target_network(self):
        """
        Copies the weights from the online network
        to the target network.
        """

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

    def train_step(self, replay_buffer):

        if len(replay_buffer) < BATCH_SIZE:
            return None

        batch = replay_buffer.sample(BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions).squeeze()

        with torch.no_grad():
            max_next_q = self.target_model(
                next_states
            ).max(1)[0]

        target_q = rewards + GAMMA * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()