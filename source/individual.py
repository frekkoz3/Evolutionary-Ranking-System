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
        - GeneticIndividual
"""
import random
from chomp import Chomp

class Individual():

    def __init__(self, init_elo = 100, id = 0):
        self.elo = init_elo
        self.id = id # this should be unique for each individual

    def get_elo(self):
        return self.elo
    
    def update_elo(self, new_elo):
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