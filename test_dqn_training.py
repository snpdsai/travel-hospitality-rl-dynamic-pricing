import random

from src.dqn_agent import DQNAgent
from src.replay_buffer import ReplayBuffer

agent = DQNAgent()
buffer = ReplayBuffer()

# Fill replay buffer with random transitions
for _ in range(100):

    state = [
        random.randint(0, 50),
        random.randint(0, 30)
    ]

    action = random.randint(0, 4)

    reward = random.randint(80, 120)

    next_state = [
        max(0, state[0] - random.randint(0, 1)),
        max(0, state[1] - 1)
    ]

    done = random.choice([True, False])

    buffer.add(
        state,
        action,
        reward,
        next_state,
        done
    )

loss = agent.train_step(buffer)

print("Replay Buffer Size:", len(buffer))
print("Training Loss:", loss)