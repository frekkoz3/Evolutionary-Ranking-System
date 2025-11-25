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


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)