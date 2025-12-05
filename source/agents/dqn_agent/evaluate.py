"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    evaluate.py

    This file contains the evaluation side for the dqn agent.
"""
from source.games.boxing.boxing import *
from source.games.console import *
from dqn_agent import *
from tqdm import tqdm
from source.agents.individual import LogicalAIIndividual, RealIndividual

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("--v1", type=int, default=1, help="player 1 checkpoint")
    parser.add_argument("--v2", type=int, default=1, help="player 2 checkpoint")
    parser.add_argument("--human", type = bool, default=False, help="if testing p1 versus an human")
    # Parse
    args = parser.parse_args()

    p1_v, p2_v = args.v1, args.v2
    human = args.human
    p1 = DQNAgent.load(f"{ROOT}/source/individuals/individual{p1_v}.pth")
    p2 = DQNAgent.load(f"{ROOT}/source/individuals/individual{p2_v}.pth")
    p3 = RealIndividual()
    if human:
        play_boxing(players=[p1, p3], render_mode="human", eval_mode = True)
    else:
        play_boxing(players=[p1, p2], render_mode="human", eval_mode = True)