"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    matchmaking.py:

    This file contains all the handler for the matchmaking between individuals.
    Functions list:
        - matches(individuals)
"""
from individual import *

def matches(individuals):
    """
        This function should provides, given all the individuals, proper matches.
        In this moment is just a random shuffle.
        This function returns a list of tuples, each of them containing two individuals.
    """

    idxs = [i for i in range (len(individuals))]
    random.shuffle(idxs)
    return [(individuals[idxs[i]], individuals[idxs[i+1]]) for i in range (0, len(idxs), 2)]

if __name__ == '__main__':
    
    pass