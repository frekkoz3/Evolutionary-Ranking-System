"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    replay_buffer.py

    This file contains the implementation of the replay buffer for the dqn.

    transition : a named tuple representing a single transition in our environment. 
    It essentially maps (state, action) pairs to their (next_state, reward) result.
"""
import random
from collections import namedtuple, deque
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
            This sample methods try to give more probability to sample states were a reward was collected (both positive and negative ones)
        """
        mem_list = list(self.memory)

        # Extract reward from each transition
        rewards = np.array([t.reward for t in mem_list], dtype=np.float64)

        # Weights are the absolute values of the rewards + 1 
        weights = np.absolute(rewards) + 1 
        weights = weights / weights.sum()

        # Weighted sample without replacement
        idxs = np.random.choice(len(mem_list), batch_size, replace=False, p=weights.flatten())
        samples = [mem_list[i] for i in idxs]

        return random.sample(self.memory, batch_size) # samples

    def __len__(self):
        return len(self.memory)