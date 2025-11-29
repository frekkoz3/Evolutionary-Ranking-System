"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    evaluate.py

    This file contains the evaluation side for the dqn agent.
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

    p1_v, p2_v = 0, 0
    p1 = DQNAgent.load(f"players/p1_{p1_v}.pth")
    p2 = DQNAgent.load(f"players/p2_{p2_v}.pth")
    play_boxing(players=[p1, p2], render_mode="human", eval_mode = True)