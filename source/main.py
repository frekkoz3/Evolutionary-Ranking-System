"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    main.py:
    
    This file contains the wrapper for the project.
"""
from ELO import *
from console import *
from individual import *
import random

if __name__ == '__main__':

    lam = 400 # lambda for the probability of winning
    k = 40 # 

    rows = 10 # number of rows on the chomp board
    cols = 10 # number of columns on the chomp board
    poison_position = [-1, -1] # position of the poisoned block on the chomp board. [-1, -1] = random

    n = 10 # number of individuals. please keep it a multiple of 2 for now

    players = [RandomIndividual() for i in range (n)]
    
    elos = [[] for i in range (n)]

    number_of_matches = 1000

    for i in range (number_of_matches):
        matches = [i for i in range (n)]
        random.shuffle(matches)
        for j in range (0, len(matches), 2):
            player_1 = players[matches[j]]
            player_2 = players[matches[j+1]]
            x = player_1.get_elo()
            y = player_2.get_elo()
            result = play_chomp(rows = rows, cols = cols, poison_position=poison_position, players = [player_1, player_2], graphics=False)
            x, y = return_function(x, y, result, k=k, lam=lam)
            player_1.update_elo(x)
            player_2.update_elo(y)
            elos[matches[j]].append(x)
            elos[matches[j+1]].append(y)
    
    from matplotlib import pyplot as plt
    
    log_log = False

    for i in range (n):
        if log_log:
            plt.plot(np.log([i + 1 for i in range (number_of_matches)]), np.log(elos[i]))#, label = f"player {i}")
        else:
            plt.plot([i + 1 for i in range (number_of_matches)], elos[i])#, label = f"player {i}")

    #plt.legend()
    if log_log:
        plt.title("LOG LOG ELO PROGRESSION")
    else:
        plt.title("ELO PROGRESSION")
    plt.show()