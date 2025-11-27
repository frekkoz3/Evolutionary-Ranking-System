"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    individual.py:
    
    This file contains the class for the individual representation.
    The mother class is the "Individual".
    Individuals have natively the attribute elo, since these individuals all will be inserted in an evolutionary ranking system.

    Classes list:
        - Individual
        - RealIndividual
        - RandomIndividual
        - GeneticPolicyIndividual
"""
import sys
import os

# Add the root of the project to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from itertools import count
from source.policy import *
from gymnasium import Env
import copy

class Individual():
    _ids = count(0)

    def __init__(self, init_elo = 100):
        self.elo = init_elo
        self.id =  next(Individual._ids)
        """self.elo_history = []
        self.opponent_history = []"""

    def get_elo(self):
        return self.elo
    
    def get_id(self):
        return self.id
    
    def update_elo(self, opponent_id, new_elo):
        """self.opponent_history.append(opponent_id)
        self.elo_history.append(self.elo)"""
        self.elo = max(new_elo, 0)

    def overwrite(self, other):
        self.__dict__ = copy.deepcopy(other.__dict__)

    def observe(self, obs, action, rew, new_obs, done):
        pass

    def move(self):
        pass

    def update(self, reward):
        pass

import pygame

class RealIndividual(Individual):

    def move(self, env):
        """
        This move function will use the env.keymap attribute.
        the keymap is a dict mapping pygame keys to actions.
        """
        keys = pygame.key.get_pressed()

        # default no-op
        action = 0

        for key, action_value in env.keymap.items():
            if keys[key]:
                return action_value
        
        return action
    
class RandomIndividual(Individual):
    
    def move(self, game : Env):
        return game.action_space.sample()

class GeneticPolicyIndividual(Individual):
    """
        This class should be the implementation of the individual we actually want to implement for the Evolutionary Ranking System project.
        It should presents all what it needs to learn, mutate etc.
        It should use the utilities from the evo_utils.py and it should use the policy as an attribute.
    """

    def __init__(self, initial_policy, init_elo = 100):
        super().__init__(init_elo)
        self.policy = initial_policy

    def move(self, game):
        state = self.policy.transform(game.get_state())
        return self.policy[state]

if __name__ == '__main__':
    
    pass