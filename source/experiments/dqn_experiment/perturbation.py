"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    perturbation.py

    This file contains an experiment for the mutation of an agent.
"""
import sys
import os

# Add the root of the project to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from source.games.boxing.boxing import *
from source.games.console import *
from source.agents.dqn_agent.dqn_agent import *
from source.agents.individual import LogicalAIIndividual, RealIndividual

import argparse


def perturb_model(model, scale=0.01):
    with torch.no_grad():            # avoid tracking in autograd
        for p in model.parameters():
            p.add_(torch.randn_like(p) * scale)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("--c1", type=int, default=1, help="player 1 checkpoint")
    parser.add_argument("--c2", type=int, default=1, help="player 2 checkpoint")
    parser.add_argument("--human", type = bool, default=False, help="if testing p1 versus an human")
    # Parse
    args = parser.parse_args()

    p1_v, p2_v = args.c1, args.c2
    human = args.human
    p1 = DQNAgent.load(f"players/p1_v4_{p1_v}.pth")
    p2 = DQNAgent.load(f"players/p1_v4_{p1_v}.pth")
    perturb_model(p2.policy_net, scale = 0.1)
    perturb_model(p2.target_net, scale=0.1)
    p3 = RealIndividual()
    if human:
        play_boxing(players=[p1, p3], render_mode="human", eval_mode = True)
    else:
        play_boxing(players=[p1, p2], render_mode="human", eval_mode = True)