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

from itertools import count
from policy import *
from gymnasium import Env
import copy

class Individual():
    _ids = count(0)

    def __init__(self, init_elo = 100):
        self.elo = init_elo
        self.id =  next(Individual._ids)
        self.elo_history = []
        self.opponent_history = []

    def get_elo(self):
        return self.elo
    
    def get_id(self):
        return self.id
    
    def update_elo(self, opponent_id, new_elo):
        self.opponent_history.append(opponent_id)
        self.elo_history.append(self.elo)
        self.elo = max(new_elo, 0)

    def overwrite(self, other):
        self.__dict__ = copy.deepcopy(other.__dict__)

    def move(self):
        pass

    def update(self, reward):
        pass

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