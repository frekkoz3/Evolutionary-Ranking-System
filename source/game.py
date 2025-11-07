"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    game.py:
    
    This file contains the handler for the entire game.
"""
import ELO as elo 
import console as cns
import individual as ind
import matchmaking as mmk
import numpy as np
from matplotlib import pyplot as plt

def play():
    """
        This function should provides the complete wrapper for everything.
        It should be configurable from a json or something like this.
    """

    # --- SETTINGS ---
    
    lam = 400 # lambda for the probability of winning
    k = 40 # 

    rows = 10 # number of rows on the chomp board
    cols = 10 # number of columns on the chomp board
    poison_position = [-1, -1] # position of the poisoned block on the chomp board. [-1, -1] = random

    n = 10 # number of individuals. please keep it a multiple of 2 for now

    players = [ind.RandomIndividual() for _ in range (n)]

    number_of_matches = 1000

    # --- ACTUAL GAMES ---

    for i in range (number_of_matches):
        matches = mmk.matches(players)
        for match in matches:
            p1, p2 = match
            x = p1.get_elo()
            y = p2.get_elo()
            result = cns.play_chomp(rows = rows, cols = cols, poison_position=poison_position, players = [p1, p2], graphics=False)
            x, y = elo.return_function(x, y, result, k=k, lam=lam)
            p1.update_elo(p2, x)
            p2.update_elo(p1, y)
    
    # --- RESULTS ---
    
    log_log = False

    for i in range (n):
        if log_log:
            plt.plot(np.log([i + 1 for i in range (number_of_matches)]), np.log(players[i].elo_history))#, label = f"player {i}")
        else:
            plt.plot([i + 1 for i in range (number_of_matches)], players[i].elo_history)#, label = f"player {i}")

    #plt.legend()
    if log_log:
        plt.title("LOG LOG ELO PROGRESSION")
    else:
        plt.title("ELO PROGRESSION")
    plt.show()

if __name__ == '__main__':
    
    pass