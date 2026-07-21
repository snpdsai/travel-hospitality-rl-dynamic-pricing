import torch

from src.dqn_agent import DQN

model = DQN()

print(model)

state = torch.tensor(
    [[50.0, 30.0]]
)

q_values = model(state)

print()

print("Q Values")

print(q_values)

print()

print("Output Shape:", q_values.shape)