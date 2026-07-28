import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.environment import DynamicPricingEnv
from src.replay_buffer import ReplayBuffer

from src.config import (
    PRICE_LEVELS,
    DQN_GAMMA,
    LEARNING_RATE,
    BATCH_SIZE,
    REPLAY_CAPACITY,
    DQN_EPSILON_START,
    DQN_EPSILON_DECAY,
    DQN_EPSILON_MIN,
    TAU,
)

from src.config import DQN_EPISODES
from src.config import EVALUATION_EPISODES

from pathlib import Path

class DQN(nn.Module):

    def __init__(
        self,
        state_size=2,
        action_size=None,
    ):

        super().__init__()

        if action_size is None:
            action_size = len(PRICE_LEVELS)

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):

        return self.network(x)


class DQNAgent:
    """
    Deep Q-Network agent for the
    Dynamic Pricing environment.
    """

    def __init__(self):

        self.env = DynamicPricingEnv()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = DQN().to(self.device)

        self.target_net = DQN().to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.memory = ReplayBuffer(REPLAY_CAPACITY)

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=LEARNING_RATE
        )

        self.criterion = nn.SmoothL1Loss()

        self.gamma = DQN_GAMMA

        self.tau = TAU

        self.batch_size = BATCH_SIZE

        self.epsilon = DQN_EPSILON_START
        self.epsilon_decay = DQN_EPSILON_DECAY
        self.epsilon_min = DQN_EPSILON_MIN

    def choose_action(
        self,
        state: np.ndarray,
    ) -> int:

        if random.random() < self.epsilon:

            return self.env.action_space.sample()

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_net(state)

        return torch.argmax(q_values).item()

    def decay_epsilon(self) -> None:

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def soft_update_target_network(self):
        """
        Soft update the target network using Polyak averaging.
        """

        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):

            target_param.data.copy_(
                self.tau * policy_param.data +
                (1.0 - self.tau) * target_param.data
            )

    def train_step(self):
        """
        Perform one optimization step using a mini-batch
        sampled from replay memory.
        """

        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        current_q = self.policy_net(states).gather(
            1,
            actions
        ).squeeze()

        with torch.no_grad():

            max_next_q = self.target_net(
                next_states
            ).max(1)[0]

        target_q = rewards + (
            1 - dones
        ) * self.gamma * max_next_q

        loss = self.criterion(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        return loss.item()

    def train(
        self,
        episodes=DQN_EPISODES,
    ):

        model_dir = Path("../models")
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.policy_net.train()

        reward_history = []
        loss_history = []

        best_average_reward = -float("inf")

        for episode in range(episodes):

            state, _ = self.env.reset()

            done = False

            total_reward = 0

            episode_losses = []

            while not done:

                action = self.choose_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated

                self.memory.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                state = next_state

                total_reward += reward

                loss = self.train_step()

                if loss is not None:
                    episode_losses.append(loss)

            self.decay_epsilon()

            reward_history.append(total_reward)

            if episode_losses:
                loss_history.append(np.mean(episode_losses))
            else:
                loss_history.append(np.nan)

            if len(reward_history) >= 20:

                recent_average = np.mean(reward_history[-20:])

                if recent_average > best_average_reward:

                    best_average_reward = recent_average

                    torch.save(
                        self.policy_net.state_dict(),
                        model_dir / "best_dqn_model.pth",
                    )

            self.soft_update_target_network()

            if (episode + 1) % 50 == 0:

                print(
                    f"Episode {episode+1}/{episodes} | "
                    f"Revenue = {total_reward} | "
                    f"Loss = {loss_history[-1]:.2f} | "
                    f"Epsilon = {self.epsilon:.3f}"
                )

                # Save the final model after training completes
        torch.save(
            self.policy_net.state_dict(),
            model_dir / "final_dqn_model.pth",
        )

        if best_average_reward == -float("inf"):
            print("\nTraining completed before 20 episodes.")
            print("No moving-average checkpoint was calculated.")
        else:
            print(f"\nBest 20-Episode Average Revenue: {best_average_reward:.2f}")
        print("Best model saved to :", model_dir / "best_dqn_model.pth")
        print("Final model saved to:", model_dir / "final_dqn_model.pth")

        return reward_history, loss_history

    def evaluate(self, episodes=EVALUATION_EPISODES):
        """
        Evaluate the trained DQN policy without exploration.
        """

        revenues = []

        original_epsilon = self.epsilon

        self.epsilon = 0.0

        self.policy_net.eval()

        for _ in range(episodes):

            state, _ = self.env.reset()

            done = False
            total_reward = 0

            while not done:

                action = self.choose_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                total_reward += reward

                state = next_state

                done = terminated or truncated

            revenues.append(total_reward)

        self.epsilon = original_epsilon

        self.policy_net.train()

        return revenues

    def simulate_booking_season(self):
        """
        Simulate one complete booking season using the
        trained DQN policy and record pricing decisions.
        """

        original_epsilon = self.epsilon

        self.epsilon = 0.0
        self.policy_net.eval()

        state, _ = self.env.reset()

        done = False

        trajectory = {
            "days_left": [],
            "inventory": [],
            "price": [],
            "revenue": [],
            "purchase": [],
        }

        while not done:

            action = self.choose_action(state)

            next_state, reward, terminated, truncated, info = self.env.step(action)

            trajectory["days_left"].append(state[1])
            trajectory["inventory"].append(state[0])
            trajectory["price"].append(info["price"])
            trajectory["revenue"].append(reward)
            trajectory["purchase"].append(info["purchase"])

            state = next_state

            done = terminated or truncated

        self.epsilon = original_epsilon
        self.policy_net.train()

        return trajectory

    def load_model(self, model_path):
        """
        Load trained model weights.
        """

        self.policy_net.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
        )

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.policy_net.eval()
        self.target_net.eval()