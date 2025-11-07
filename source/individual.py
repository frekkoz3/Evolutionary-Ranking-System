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
import random
from chomp import Chomp
from itertools import count
from policy import *

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
    
    def update_elo(self, opponent, new_elo):
        self.opponent_history.append(opponent.get_id())
        self.elo_history.append(self.elo)
        self.elo = new_elo

    def move(self):
        pass

class RealIndividual(Individual):

    def move(self, game : Chomp):
        move = None
        while move not in game.valid_moves():
            try:
                r = int(input("Row (0-indexed): "))
                c = int(input("Col (0-indexed): "))
                move = (r, c)
            except:
                move = None
        return move

class RandomIndividual(Individual):
    
    def move(self, game : Chomp):
        return random.choice(game.valid_moves())
    
class GeneticPolicyIndividual(Individual):

    def __init__(self, initial_policy, init_elo = 100):
        super().__init__(init_elo)
        self.policy = initial_policy

    def move(self, game : Chomp):
        state = self.policy.transform(game.get_state())
        return self.policy[state]
    
if __name__ == '__main__':
    
    pass