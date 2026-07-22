import random
from collections import deque

from src.config import BUFFER_SIZE


class ReplayBuffer:

    def __init__(self):

        self.buffer = deque(maxlen=BUFFER_SIZE)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def sample(self, batch_size):

        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(self):

        return len(self.buffer)