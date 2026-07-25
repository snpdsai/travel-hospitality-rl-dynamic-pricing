import torch

from src.dqn_agent import DQNAgent

agent = DQNAgent()

same_before = True

for p1, p2 in zip(
    agent.model.parameters(),
    agent.target_model.parameters()
):
    if not torch.equal(p1, p2):
        same_before = False
        break

print("Initially synchronized:", same_before)

# Modify the online network
with torch.no_grad():
    for param in agent.model.parameters():
        param.add_(1.0)
        break

same_after_change = True

for p1, p2 in zip(
    agent.model.parameters(),
    agent.target_model.parameters()
):
    if not torch.equal(p1, p2):
        same_after_change = False
        break

print("Same after modifying online model:", same_after_change)

# Synchronize
agent.update_target_network()

same_after_sync = True

for p1, p2 in zip(
    agent.model.parameters(),
    agent.target_model.parameters()
):
    if not torch.equal(p1, p2):
        same_after_sync = False
        break

print("Same after synchronization:", same_after_sync)