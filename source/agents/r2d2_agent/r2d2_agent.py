"""
    Developer : Bredariol Francesco

    r2d2_agent.py

    This file contains the implementation of an r2d2 agent.
"""
from source.agents.individual import Individual

import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from source.agents.r2d2_agent.replay_buffer import ReplayMemory, Transition

