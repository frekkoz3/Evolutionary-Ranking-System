"""
    Developer : Bredariol Francesco

    individual.py:
    
    This file contains the class for the individual representation.
    The mother class is the "Individual".
    Individuals have natively the attribute elo, since these individuals all will be inserted in an evolutionary ranking system.

    Classes list:
        - Individual
        - RealIndividual
        - RandomIndividual
"""

from itertools import count
from gymnasium import Env
import copy
import pygame

class Individual():
    _ids = count(0)

    def __init__(self, init_elo = 100):
        self.init_elo = init_elo
        self.elo = init_elo
        self.id =  next(Individual._ids)
        self.n_matches = 0
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
        self.n_matches += 1
    
    def reset_elo(self):
        self.elo = self.init_elo

    def overwrite(self, other):
        self.__dict__ = copy.deepcopy(other.__dict__)

    def need_map(self):
        return False # this is a flag passed to tell if a map representation of the state is needed

    def observe(self, obs, action, rew, new_obs, done, **kwargs):
        pass

    def move(self, **kwargs):
        pass

    def reset(self, **kwargs):
        pass

    def update(self, **kwrags):
        pass

    def save(self):
        pass

    def __str__(self):
        return f"{self.id} : {self.elo}"

    @classmethod
    def load(cls):
        pass

class RealIndividual(Individual):

    def move(self, env, **kwargs):
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
    
    def move(self, game : Env, **kwargs):
        return game.action_space.sample()

if __name__ == '__main__':
    
    pass