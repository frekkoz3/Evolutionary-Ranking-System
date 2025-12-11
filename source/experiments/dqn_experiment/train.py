"""
    Developer : Bredariol Francesco

    train.py

    This file contains the train side for the dqn agent.
"""

from source.games.boxing.boxing import *
from source.games.grab_n_go.grab_n_go import *
from source.games.console import *
from tqdm import tqdm
from source.agents.dqn_agent.dqn_agent import * 
from source.agents.grab_n_go_dqn_agent.gng_dqn_agent import *
import random
from source.experiments.dqn_experiment import DQN_PLAYERS_ROOT
import os

if __name__ == '__main__':
    # --------------------------------------
    # Agents definition
    # --------------------------------------
    render_mode = "non-human"
    env = BoxingEnv(render_mode)
    obs, _ = env.reset()

    name = "boxing"
    
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    checkpoint_index = 0
    """p1 = DQNAgent.load(os.path.join(DQN_PLAYERS_ROOT, f"p1_{name}_{checkpoint_index}.pth")) #(n_actions, n_observations)
    p2 = DQNAgent.load(os.path.join(DQN_PLAYERS_ROOT, f"p2_{name}_{checkpoint_index}.pth"))#(n_actions, n_observations)
    p1.reset(percentage = 0.75)
    p2.reset(percentage = 0.75)"""
    p1 = DQNAgent(n_actions, n_observations)
    p2 = DQNAgent(n_actions, n_observations)
    p1.save(os.path.join(DQN_PLAYERS_ROOT, f"p1_{name}_{0}.pth"))
    p2.save(os.path.join(DQN_PLAYERS_ROOT, f"p2_{name}_{0}.pth"))
    # --------------------------------------
    # Main loop
    # --------------------------------------
    number_of_iterations = 20
    for i in range (number_of_iterations):
        n_games = 200
        # --------------------------------------
        # Single checkpoint
        # --------------------------------------
        for j in tqdm(range (n_games), desc=f"Iteration {i} on going", unit="game"):
            p1.update_t = 0
            p2.update_t = 0
            play_boxing(players=[p1, p2], render_mode=render_mode, eval_mode= False)
            play_boxing(players=[p2, p1], render_mode=render_mode, eval_mode= False)
        p1.reset()
        p2.reset()
        # --------------------------------------
        # Checkpoints saving
        # --------------------------------------
        p1.save(os.path.join(DQN_PLAYERS_ROOT, f"p1_{name}_{checkpoint_index + 1 + i}.pth"))
        p2.save(os.path.join(DQN_PLAYERS_ROOT, f"p2_{name}_{checkpoint_index + 1 + i}.pth"))