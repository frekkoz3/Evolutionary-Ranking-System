"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    elo_system.py:
    
    This file contains the handler for the entire game using the elo_system.
"""
import source.elo_system.ELO as elo 
from source.games.console import *
import source.elo_system.matchmaking as mmk
from source.elo_system.evo_utils import *

from source.agents import *

from matplotlib import pyplot as plt
from tqdm import tqdm

from pathlib import Path
import json

from joblib import Parallel, delayed

from source import INDIVIDUALS_DIR
import os

class EES():
    """
        Evolutionary Elo System wrapper
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.config()

    def config(self):
        """
            This function provide an handler for the configuration via config.json.
            This config should be able to be consistent with the game played.
        """
        with open(self.config_path, "r") as f:
            self.configs = json.load(f)

        self.lam = self.configs["lam"] # lambda for the probability of winning
        self.min_k = self.configs["min_k"] # minimum k for the constant in the elo update
        self.max_k = self.configs["max_k"] # maximum k for the constant in the elo update
        self.t_k = self.configs["t_k"] # k for the tournament selection or for the match selection
        self.elitism = self.configs["elitism"] # number of best individuals to keep
        self.parallel = self.configs["parallel"] # setting for the parallel  -> parallel is not working rn with the gpu
        self.n_individuals = self.configs["n_individuals"] # number of individuals. please keep it a multiple of 2 for now
        self.n_seasons = self.configs["n_seasons"] # number of seasons to simulate
        self.n_rounds = self.configs["n_rounds"] # number of rounds per season

    def play(self, player_class = RandomIndividual, matchmaking_fun = mmk.matches, play_fun = play_boxing, eval_mode = False, **kwargs):
        """
            This function should provides the complete wrapper for everything.
            It should be configurable from a json or something like this.
        """

        # --- Player class specific settings ---

        if player_class == DQNAgent:
            env = kwargs["env"]
            players = [DQNAgent(n_actions = env.action_space.n, n_observations = env.observation_space.shape[0]) for _ in range (self.n_individuals)]
        elif player_class == GNGDQNAgent:
            env = kwargs["env"]
            players = [GNGDQNAgent(DQNAgent(n_actions = env.action_space.n, n_observations = env.observation_space.shape[0]), DQNAgent(n_actions = env.action_space.n, n_observations = env.observation_space.shape[0])) for _ in range (self.n_individuals)]
        elif player_class == GNGTreeAgent:
            env = kwargs["env"]
            players = [GNGTreeAgent(TreeAgent(100, env.action_space.n), TreeAgent(100, env.action_space.n)) for _ in range(self.n_individuals)]
        else:
            players = [player_class() for _ in range (self.n_individuals)]

        render_mode = "non-human"

        # --- ACTUAL GAME ---

        k_step = (self.max_k - self.min_k) / self.n_seasons

        for season in range (442, self.n_seasons):
            # --- ONLINE OPTIMIZATION (RL or whatever) --  
            for r in tqdm(range (self.n_rounds), desc=f"Season {season + 1} | {self.n_seasons}", unit="round"):
                if self.parallel:
                    parallel_round(players = players, matchmaking_fun = matchmaking_fun, play_fun = play_fun, render_mode = render_mode, eval_mode = eval_mode, k = int(k_step*season), lam = self.lam, n_jobs=-1, **kwargs) #fixed to used the maximum number of jobs
                else:
                    round(players = players, matchmaking_fun = matchmaking_fun, play_fun = play_fun, render_mode = render_mode, eval_mode = eval_mode, k = int(k_step*season), lam = self.lam, **kwargs)
            
            # --- SAVING INDIVIDUALS --- 
            print("Saving and mutating individuals...")
            for player in players:
                player.save(os.path.join(INDIVIDUALS_DIR, f"{season}_{player.id}.pth"))
            print("Individuals saved")

            # --- OFFLINE OPTIMIZATION (evolutionary strategy) ---
            new_players = sorted(players, key = lambda x : x.elo, reverse = True)[:self.elitism] # elitism is the number of best individuals to keep
            
            for _ in range (self.n_individuals - self.elitism):
                # tplayer = match_selection(players, play_fun, t_k) # this selection is based on an empirical montecarlo approach
                tplayer = tournament_selection(players, self.t_k)
                player = tplayer.__class__.load(os.path.join(INDIVIDUALS_DIR, f"{season}_{tplayer.id}.pth"))
                player.mutate()
                new_players.append(player)

            if season >= 1:
                
                folder = Path(INDIVIDUALS_DIR)
                for file_path in folder.iterdir():  # non-recursive
                    if file_path.is_file() and file_path.name.startswith(f"{season-1}_"):
                        file_path.unlink()

            if season != self.n_seasons - 1:
                players = new_players
                for i, p in enumerate(players): # resetting the elos at the end of a season
                    players[i].elo = 100

        # --- END ---
        print("It has been a pleasure, bye!")
        return players

def run_match(p1, p2, play_fun, render_mode, eval_mode, k, lam, **kwargs):

    x = p1.get_elo()
    y = p2.get_elo()
    
    result1 = play_fun(players=[p1, p2], render_mode=render_mode, eval_mode = eval_mode, **kwargs)
    result2 = play_fun(players=[p2, p1], render_mode=render_mode, eval_mode = eval_mode, **kwargs)
    # since they are zero sum games it makes sense to make players play both the sides
    result = (0, 0) if result1 == result2 else (1, 0)
    result = (0, 1) if result1[1] + result[0] == 2 else result
    x_new, y_new = elo.return_function(x, y, result, k=k, lam=lam)

    p1.update_elo(p2.get_id(), x_new)
    p2.update_elo(p1.get_id(), y_new)

    return (p1, p2)

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

def round(players : list[Individual], matchmaking_fun, play_fun, render_mode : str, eval_mode : bool, k : int, lam : int, **kwargs):
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
        result1 = play_fun(players=[p1, p2], render_mode=render_mode, eval_mode = eval_mode, **kwargs)
        result2 = play_fun(players=[p2, p1], render_mode=render_mode, eval_mode = eval_mode, **kwargs)
        # since they are zero sum games it makes sense to make players play both the sides
        result = (0, 0) if result1 == result2 else (1, 0)
        result = (0, 1) if result1[1] + result[0] == 2 else result
        x, y = elo.return_function(x, y, result, k=k, lam=lam)
        p1.update_elo(p2.get_id(), x)
        p2.update_elo(p1.get_id(), y)

def show_results(players : list[Individual], play_fun, prefix = 449, **kwargs):
    """
        This function simply show the final elo of all the individuals.
    """
    if players == None:
        players = []
        for filename in os.listdir(INDIVIDUALS_DIR):
            if filename.startswith(prefix):
                players.append(GNGTreeAgent.load(os.path.join(INDIVIDUALS_DIR, filename)))

    elos = []
    players.sort(key = lambda x : x.elo, reverse=True)
    for player in players:
        elos.append(player.elo)
    plt.hist(elos, bins = 10)
    plt.title("ELO histogram")
    plt.show() 
    top = players[0]
    try:
        top.visualize()
        top.view_probs()
    except Exception as e:
        pass
    play_fun(players=[players[0], RealIndividual()], render_mode="human", eval_mode = True, **kwargs)
    play_fun(players=[RealIndividual(), players[0]], render_mode="human", eval_mode = True, **kwargs)
    
if __name__ == '__main__':

    pass