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

    lam = 10
    k = 50

    x = 1200
    y = 1000
    
    elo_x = [x]
    elo_y = [y]
    import random
    for i in range (100):
        p_x, p_y = compute_winning_probability(x, y, lam=lam)
        print(f"player 1 elo : {x}. player 2 elo : {y}. P(X = 1|Y) = {p_x}. P(Y = 1 | X) = {p_y}")
        result = (1, 0) if random.uniform(0, 1) < p_x else (0, 1)
        x, y = return_function(x, y, result, k=k, lam=lam)
        elo_x.append(x)
        elo_y.append(y)
    
    from matplotlib import pyplot as plt

    plt.plot([i for i in range (len(elo_x))], elo_x, c = "red", label = "player 1")
    plt.plot([i for i in range (len(elo_y))], elo_y, c = "green", label = "player 2")
    plt.legend()
    plt.title("ELO PROGRESSION")
    plt.show()