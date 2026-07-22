from src.replay_buffer import ReplayBuffer

buffer = ReplayBuffer()

for i in range(100):

    buffer.add(
        state=[i, 30],
        action=i % 5,
        reward=i * 10,
        next_state=[i - 1, 29],
        done=False
    )

print("Replay Buffer Size:", len(buffer))

batch = buffer.sample(5)

print()

print("Random Batch:")

for transition in batch:
    print(transition)