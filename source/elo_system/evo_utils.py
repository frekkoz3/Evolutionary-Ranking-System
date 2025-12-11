"""
    Developer : Bredariol Francesco

    evo_utils.py:
    
    This file contains some utils for the elo_system.
    Functions list:
        - tournament_selection
        - match_selection
"""
import random 
from source.agents.individual import *

def tournament_selection(players : list[Individual], k : int = 5):
    sam = random.sample(players, k)
    return max(sam, key = lambda x : x.elo)

def match_selection(players : list[Individual], play_fun, k : int = 5, **kwargs):
    sam = random.sample(players, 2)
    r1, r2 = 0, 0
    for _ in range (k):
        results = play_fun(players = sam, render_mode = "non-human", eval_mode = True, **kwargs)
        r1 += results[0]
        r2 += results[1]
    if r1 > r2:
        return sam[0]
    if r2 > r1:
        return sam[1]
    return sam[0] if random.random() < 0.5 else sam[1]
