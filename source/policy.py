"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    policy.py:
    
    This file contains the class for the policy representation.
    Each policy takes the game state as input, transform it (if necessary) and respond with an action.
    A policy coould also has a buffer memory.
    The list of possible action are obtained by the state of the game (its something the policy must do by itself)

    Classes list:
        - Policy
"""
import random
import numpy as np

class Policy():

    def __init__(self, state):
        self.buffer_memory = []
        self.last_decision = None
        self.decision = None
        self.policy_grid = np.zeros(shape = np.array(state).shape) # this should contains the actual policy in a grid policy. this is not correct

    def transform(self, state):
        """
            Transform could be any type of transofrmation of the state.
            It could be binning, it could be differentation between different state etc.
            The state is now passed as the board grid of the game.
        """
        return state
    
    def __call__(self, state):
        """
            Return the decision for a given state. 
            This is just a random one.
        """
        state = self.transform(state)
        self.buffer_memory.append(state)
        moves = []
        rows = len(state)
        cols = len(state[0])
        for r in range(rows):
            for c in range(cols):
                if self.board[r][c] == 1:
                    moves.append((r, c))
        return random.choice(moves)
    
    def update(self, reward):
        """
            This method is designed in order to tune the policy when a reward (positive or negative) is obtained.
            It is not constrained to be an RL update method, it could also be a GP method or whatever.
        """
        self.policy_grid[self.last_decision] += reward # this is completely wrong, just to put some code here