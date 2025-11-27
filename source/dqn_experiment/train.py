"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    train.py

    This file contains the train side for the dqn agent.
"""
import sys
import os

# Add the root of the project to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from source.games.boxing.boxing import *
from source.console import *
from dqn_agent import *
from tqdm import tqdm

if __name__ == '__main__':

    render_mode = "non-human"
    env = BoxingEnv(render_mode)
    obs, _ = env.reset()
    n_observations = len(obs) + 1 # the additional one is the player side one
    n_actions = env.action_space.n
    p1 = DQNAgent(n_actions, n_observations)
    p2 = DQNAgent(n_actions, n_observations)
    n_games = 100
    for i in tqdm(range (n_games), desc="Games on going", unit="games"):
        if i%2 == 0:
            play_boxing(players=[p1, p2], render_mode=render_mode)
        else:
            play_boxing(players=[p2, p1], render_mode=render_mode)
    render_mode = "human"
    env = BoxingEnv(render_mode)
    play_boxing(players=[p1, p2], graphics=True)