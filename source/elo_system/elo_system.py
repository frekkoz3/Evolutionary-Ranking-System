"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    elo_system.py:
    
    This file contains the handler for the entire game using the elo_system.
"""
import source.elo_system.ELO as elo 
import source.games.console as cns
import source.agents.individual as ind
from source.agents.dqn_agent.dqn_agent import DQNAgent
from source.agents.grab_n_go_dqn_agent.gng_dqn_agent import GNGDQNAgent
from source.agents.grab_n_go_tree_agent.gng_tree_agent import GNGTreeAgent
from source.agents.tree.tree import TreeAgent
import source.elo_system.matchmaking as mmk
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from source.elo_system.evo_utils import *

from source import INDIVIDUALS_DIR
import os

def config():
    """
        This function provide an handler for the configuration via config.json.
        This config should be able to be consistent with the game played.
    """

def run_match(p1, p2, play_fun, render_mode, eval_mode, k, lam, **kwargs):

    x = p1.get_elo()
    y = p2.get_elo()
    
    result = play_fun(players=[p1, p2], render_mode=render_mode, eval_mode = eval_mode, **kwargs)
    x_new, y_new = elo.return_function(x, y, result, k=k, lam=lam)

    p1.update_elo(p2.get_id(), x_new)
    p2.update_elo(p1.get_id(), y_new)

    return (p1, p2)

from joblib import Parallel, delayed

def parallel_round(players, matchmaking_fun, play_fun, render_mode : str, eval_mode : bool, k : int, lam : int, n_jobs=-1, **kwargs):
    """
        This function provide an handler for playing all the rounds once selected the matches in parallel. 
        Please note that one could actually change the game just changing the play_fun.
        It is just imposed that the play function uses the graphics option (true / false).
        The k and lam are used from the elo updater.
    """
    matches = matchmaking_fun(players)

    players_map = {player.id : player for player in players}

    results = Parallel(n_jobs=n_jobs)( 
        delayed(run_match)(p1, p2, play_fun, render_mode, eval_mode, k, lam, **kwargs)
        for (p1, p2) in matches
    ) # this works on copy of the actual players so now we have to copy them back
    # we could think of computing just the result in parallel and then update the guys serially

    # Apply the updates to the real objects
    for p1, p2 in results:
        players_map[p1.get_id()].overwrite(p1)
        players_map[p2.get_id()].overwrite(p2)

def round(players : list[ind.Individual], matchmaking_fun, play_fun, render_mode : str, eval_mode : bool, k : int, lam : int, **kwargs):
    """
        This function provide an handler for playing all the rounds once selected the matches. 
        Please note that one could actually change the game just changing the play_fun.
        It is just imposed that the play function uses the graphics option (true / false).
        The k and lam are used from the elo updater.
    """

    matches = matchmaking_fun(players)

    for match in matches:
        p1, p2 = match if random.random() < 0.5 else (match[1], match[0])
        x = p1.get_elo()
        y = p2.get_elo()
        result = play_fun(players = [p1, p2], render_mode = render_mode, eval_mode = eval_mode, **kwargs)
        x, y = elo.return_function(x, y, result, k=k, lam=lam)
        p1.update_elo(p2.get_id(), x)
        p2.update_elo(p1.get_id(), y)

def play(player_class = ind.RandomIndividual, matchmaking_fun = mmk.matches, play_fun = cns.play_boxing, parallel = False, eval_mode = False, **kwargs):
    """
        This function should provides the complete wrapper for everything.
        It should be configurable from a json or something like this.
    """

    # --- SETTINGS ---

    # lam, k, n, players, number_of_matches, kwargs = config() 
    
    lam = 400 # lambda for the probability of winning
    k = 20 # k for the constant in the elo update

    t_k = 5 # k for the tournament selection or for the match selection

    elitism = 5 # number of best individuals to keep

    parallel = True # setting for the parallel  -> parallel is not working rn with the gpu

    n = 100 # number of individuals. please keep it a multiple of 2 for now

    if player_class == DQNAgent:
        env = kwargs["env"]
        players = [DQNAgent(n_actions = env.action_space.n, n_observations = env.observation_space.shape[0]) for _ in range (n)]
    elif player_class == GNGDQNAgent:
        env = kwargs["env"]
        players = [GNGDQNAgent(DQNAgent(n_actions = env.action_space.n, n_observations = env.observation_space.shape[0]), DQNAgent(n_actions = env.action_space.n, n_observations = env.observation_space.shape[0])) for _ in range (n)]
    elif player_class == GNGTreeAgent:
        env = kwargs["env"]
        players = [GNGTreeAgent(TreeAgent(100, env.action_space.n), TreeAgent(100, env.action_space.n)) for _ in range(n)]
    else:
        players = [player_class() for _ in range (n)]

    number_of_seasons = 5

    number_of_rounds = 200

    render_mode = "non-human"

    # --- ACTUAL GAME ---

    for season in range (number_of_seasons):
        # --- ONLINE OPTIMIZATION (RL or whatever) --  
        for r in tqdm(range (number_of_rounds), desc="Tournament on going", unit="round"):
            if parallel:
                parallel_round(players = players, matchmaking_fun = matchmaking_fun, play_fun = play_fun, render_mode = render_mode, eval_mode = eval_mode, k = k, lam = lam, n_jobs=-1, **kwargs) #fixed to used the maximum number of jobs
            else:
                round(players = players, matchmaking_fun = matchmaking_fun, play_fun = play_fun, render_mode = render_mode, eval_mode = eval_mode, k = k, lam = lam, **kwargs)
        
        # --- SAVING INDIVIDUALS --- 
        print("Saving and mutating individuals...")
        for player in players:
            player.save(os.path.join(INDIVIDUALS_DIR, f"{season}_{player.id}.pth"))
            print(f"{player.id} : {player.elo}")
        print("Individuals saved")

        # --- OFFLINE OPTIMIZATION (evolutionary strategy) ---
        new_players = sorted(players, key = lambda x : x.elo, reverse = True)[:elitism] # elitism is the number of best individuals to keep
        
        for _ in range (n - elitism):
            # tplayer = match_selection(players, play_fun, t_k) # this selection is based on an empirical montecarlo approach
            tplayer = tournament_selection(players, t_k)
            player = tplayer.__class__.load(os.path.join(INDIVIDUALS_DIR, f"{season}_{tplayer.id}.pth"))
            player.mutate()
            new_players.append(player)
        
        if season != number_of_seasons - 1:
            players = new_players
            for i, p in enumerate(players): # resetting the elos at the end of a season
                players[i].elo = 100

    # --- END ---
    print("It has been a pleasure, bye!")
    return players

def show_results(path = "individuals/", starting_index = 0, number_of_individuals = 40):
    """
        This function simply show the final elo of all the individuals.
    """
    elos = []
    for n in range(starting_index, starting_index + number_of_individuals):
        t = DQNAgent.load(f"{path}individual{n}.pth")
        elos.append(t.elo)
        print(f"Individual {t.id}:\n ELO reached : {t.elo}")
    from matplotlib import pyplot as plt
    plt.hist(elos, bins = 10)
    plt.title("ELO histogram")
    plt.show() 
    
if __name__ == '__main__':

    # rows = 10 # number of rows on the chomp board
    # cols = 10 # number of columns on the chomp board
    # poison_position = [-1, -1] # position of the poisoned block on the chomp board. [-1, -1] = random
    
    # kwargs = { "rows" : rows, "cols" : cols, "poison_position" : poison_position}
    show_results(starting_index=0, number_of_individuals=60)