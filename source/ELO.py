"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco
"""
import numpy as np
import math

def compute_winning_probability(elo_x, elo_y, lam : int = 400):
    """
         The prior winning probability is computed as follow:

         P(x winning | y its opponent) = 1 / (1 + e^(ELO(Y) - ELO(X))/lam)

         This probability is driven by the lamda (lam) factor.
    """
    assert isinstance(lam, int)

    p_x = 1/(1 + math.pow(math.e, (elo_y - elo_x) / lam))
    p_y = 1/(1 + math.pow(math.e, (elo_x - elo_y) / lam))
    
    return p_x, p_y

def return_function(elo_x, elo_y, result, k : int = 20, lam : int = 400):
    """
        The return function retusns the gain or the loss for each individuals given the actual result of a match.

        Parameters:
        - elo_x (int) : elo level of the first individual
        - elo_y (int) : elo level of the secondo individual
        - result (tuple(int)) : tuple representing the outcome of the match.
                                (1, 0) means that the first individual won, (0, 1) means that the second individual won
        - k (int) : k to use in the update elo function
        - lam (int) : lambda to use in the compute_winning_probability function 

        The update ELO function is the following:

        given p_x, p_y = winnin_prob(x, y, lam)
        
        ELO (X) <- ELO(X) + k*(result[x] - p_x)
        ELO (Y) <- ELO(Y) + k*(result[y] - p_y)
    """
    assert result == (1, 0) or result == (0, 1)
    assert isinstance(k, int)
    assert isinstance(lam, int)
    
    p_x , p_y = compute_winning_probability(elo_x, elo_y, lam)
    new_elo_x = elo_x + k*(result[0] - p_x)
    new_elo_y = elo_y + k*(result[0] - p_y)

    return new_elo_x, new_elo_y

if __name__ == '__main__':

    lam = 200
    k = 20

    n = 2

    import random

    players = [random.randrange(800, 1200) for i in range (n)]
    
    elos = [[] for i in range (n)]

    number_of_matches = 100

    for i in range (number_of_matches):
        matches = [i for i in range (n)]
        random.shuffle(matches)
        for j in range (0, len(matches), 2):
            x = players[matches[j]]
            y = players[matches[j+1]]
            p_x, p_y = compute_winning_probability(x, y, lam)
            u = random.uniform(0, 1)
            result = (1, 0) if u < p_x else (0, 1)
            print(f"{u} : {result}")
            x, y = return_function(x, y, result, k=k, lam=lam)
            elos[matches[j]].append(x)
            elos[matches[j+1]].append(y)
    
    from matplotlib import pyplot as plt

    for i in range (n):
        plt.plot([i for i in range (number_of_matches)], elos[i], label = f"player {i}")

    plt.legend()
    plt.title("ELO PROGRESSION")
    plt.show()