from collections import deque
import random

import numpy as np


class ReplayBuffer:
    """
    A buffer that stores experiences for experience replay.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        """
        Append given experience to the buffer.
        
        Parameters
        ----------
        state : list of float
            The current state s given by the environment.
        action : int
            Action a chosen by the agent.
        reward : float
            Reward given by the environment for given state s and action a.
        next_state : list of float
            The resulting state s' after taking action a on state s.
        done: bool
            True if the episode has terminated.
        """
        state      = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Uniformly sample a batch from the buffer.

        Parameters
        ----------
        batch_size : int
            The size of the batch to sample.

        Returns
        -------
        batch : tuple
            The batch of samples.
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
