"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    elo_system.py:
    
    This file contains the handler for the entire game using the elo_system.
"""
import sys
import os

# Add the root of the project to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import source.ELO as elo 
import source.console as cns
import source.individual as ind
import source.matchmaking as mmk
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def show_results(players : list[ind.Individual], log_log = False):
    """
        This function shows results in terms of elo obtained by all the players.
    """
    number_of_matches = len(players[0].elo_history)

    for player in players:
        if log_log:
            plt.plot(np.log([i + 1 for i in range (number_of_matches)]), np.log(player.elo_history))#, label = f"player {i}")
        else:
            plt.plot([i + 1 for i in range (number_of_matches)], player.elo_history)#, label = f"player {i}")

    #plt.legend()
    if log_log:
        plt.title("LOG LOG ELO PROGRESSION")
    else:
        plt.title("ELO PROGRESSION")
    plt.show()

def config():
    """
        This function provide an handler for the configuration via config.json.
        This config should be able to be consistent with the game played.
    """

def run_match(p1, p2, play_fun, graphics, k, lam, kwargs):

    x = p1.get_elo()
    y = p2.get_elo()
    
    result = play_fun(players=[p1, p2], graphics=graphics, **kwargs)
    
    x_new, y_new = elo.return_function(x, y, result, k=k, lam=lam)

    p1.update_elo(p2.get_id(), x_new)
    p2.update_elo(p1.get_id(), y_new)

    return (p1, p2)

from joblib import Parallel, delayed

def parallel_round(players, matchmaking_fun, play_fun, graphics, k, lam, n_jobs=-1, **kwargs):
    """
        This function provide an handler for playing all the rounds once selected the matches in parallel. 
        Please note that one could actually change the game just changing the play_fun.
        It is just imposed that the play function uses the graphics option (true / false).
        The k and lam are used from the elo updater.
    """

    matches = matchmaking_fun(players)

    players_map = {player.id : player for player in players}

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_match)(p1, p2, play_fun, graphics, k, lam, kwargs)
        for (p1, p2) in matches
    ) # this works on copy of the actual players so now we have to copy them back

    # Apply the updates to the real objects
    for p1, p2 in results:
        players_map[p1.get_id()].overwrite(p1)
        players_map[p2.get_id()].overwrite(p2)

def round(players : list[ind.Individual], matchmaking_fun,  play_fun, graphics : bool, k : int, lam : int, **kwargs):
    """
        This function provide an handler for playing all the rounds once selected the matches. 
        Please note that one could actually change the game just changing the play_fun.
        It is just imposed that the play function uses the graphics option (true / false).
        The k and lam are used from the elo updater.
    """

    matches = matchmaking_fun(players)

    for match in matches:
        p1, p2 = match
        x = p1.get_elo()
        y = p2.get_elo()
        result = play_fun(players = [p1, p2], graphics = graphics, **kwargs)
        x, y = elo.return_function(x, y, result, k=k, lam=lam)
        p1.update_elo(p2.get_id(), x)
        p2.update_elo(p1.get_id(), y)

def play(player_class = ind.RandomIndividual, matchmaking_fun = mmk.matches, play_fun = cns.play_boxing, parallel = True, **kwargs):
    """
        This function should provides the complete wrapper for everything.
        It should be configurable from a json or something like this.
    """

    # --- SETTINGS ---

    # lam, k, n, players, number_of_matches, kwargs = config() 
    
    lam = 400 # lambda for the probability of winning
    k = 40 # k for the constant in the elo update

    n = 20 # number of individuals. please keep it a multiple of 2 for now

    players = [player_class() for _ in range (n)]

    number_of_rounds = 1000

    # --- ACTUAL GAME ---
    # please note that the actual game played could be anything. It should be sufficient to change the play_fun 

    for r in tqdm(range (number_of_rounds), desc="Tournament on going", unit="round"):
        if parallel:
            parallel_round(players = players, matchmaking_fun= matchmaking_fun, play_fun = play_fun, graphics = False, k = k, lam = lam, n_jobs=-1, **kwargs) #fixed to used the maximum number of jobs
        else:
            round(players = players, matchmaking_fun= matchmaking_fun, play_fun = play_fun, graphics = False, k = k, lam = lam, **kwargs)
        
    # --- RESULTS ---
    
    show_results(players)

if __name__ == '__main__':

    # rows = 10 # number of rows on the chomp board
    # cols = 10 # number of columns on the chomp board
    # poison_position = [-1, -1] # position of the poisoned block on the chomp board. [-1, -1] = random
    
    # kwargs = { "rows" : rows, "cols" : cols, "poison_position" : poison_position}
    pass