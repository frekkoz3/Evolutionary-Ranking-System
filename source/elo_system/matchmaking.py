"""
    Developer : Bredariol Francesco

    matchmaking.py:

    This file contains all the handler for the matchmaking between individuals.
    Functions list:
        - matches(individuals)
"""
import random

def matches(individuals):
    """
        This function should provides, given all the individuals, proper matches.
        In this moment is just a random shuffle over k ordered (elo based) bins (k is 5 if len(individuals) > 10, otherwise are just ordered couples)
        This function returns a list of tuples, each of them containing two individuals.
    """
    idxs = sorted(range(len(individuals)), key=lambda i: individuals[i].elo)
    if len(individuals) > 10:
        out = []
        i = 0
        sizes = [len(individuals)//5 if j != 4 else len(individuals)//5 + len(individuals)%5 for j in range(0, 5)]
        for size in sizes:
            part = idxs[i:i+size]
            random.shuffle(part)
            out.extend(part)
            i += size
        return [(individuals[out[j]], individuals[out[j+1]]) for j in range (0, len(out), 2)]
        
    return [(individuals[idxs[i]], individuals[idxs[i+1]]) for i in range (0, len(idxs), 2)]

if __name__ == '__main__':
    
    pass