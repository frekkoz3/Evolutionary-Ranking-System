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
from source.individual import LogicalAIIndividual

if __name__ == '__main__':
    # --------------------------------------
    # Agents definition
    # --------------------------------------
    render_mode = "non-human"
    env = BoxingEnv(render_mode)
    obs, _ = env.reset()
    n_observations = len(obs)
    n_actions = env.action_space.n
    p1 = DQNAgent(n_actions, n_observations)
    p2 = DQNAgent(n_actions, n_observations)
    # --------------------------------------
    # Main loop
    # --------------------------------------
    number_of_iterations = 3
    for j in range (number_of_iterations):
        n_games = 500
        render_mode = "non-human"
        # --------------------------------------
        # Single checkpoint
        # --------------------------------------
        p1.reset()
        p2.reset()
        for i in tqdm(range (n_games), desc=f"Iteration {j} on going", unit="game"):
            p1.update_t = 0
            p2.update_t = 0
            if i%2 == 0:
                play_boxing(players=[p1, p2], render_mode=render_mode, eval_mode= False)
            else:
                play_boxing(players=[p2, p1], render_mode=render_mode, eval_mode= False)
        render_mode = "human"
        play_boxing(players=[p1, p2], render_mode="human", eval_mode=True)
        # --------------------------------------
        # Checkpoints saving
        # --------------------------------------
        p1.save(path = f"p1_{j}.pth")
        p2.save(path = f"p2_{j}.pth")