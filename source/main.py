"""
    Developer : Bredariol Francesco

    main.py:
    
    This file is used just as final wrapper.
"""

from source.games.console import *
from source.agents import *
from source.games import *

from source.elo_system.elo_system import EES, show_results

from source import CONFIG_DIR, PROJECT_ROOT
import os

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--test", type=bool, default=True, help="test mode")
    parser.add_argument("--train", type=bool, default=False, help="train mode")
    parser.add_argument("--h", type=bool, default=False, help="help message")

    # Parse
    args = parser.parse_args()

    if args.test:
        print("Now you will play against the best Tree-individual found so far in the Grab And Go environment.\nYou will play 2 games.\nIn the first game you will be the runner, in the second game you will be the catcher.\nTo play use the 'WASD' keys.\nGood Luck (Nah, just joking, the individual is so bad).")
        show_results(players=None, play_fun=play_grab_n_go, prefix="499_", **{"env" : GrabNGoEnv(render_mode="non-human")})
    elif args.train:
        ees = EES(config_path=os.path.join(CONFIG_DIR, "config.json"))
        #players = ees.play(player_class=DQNAgent, play_fun=play_boxing, **{"env" : BoxingEnv(render_mode="non-human")})
        players = ees.play(player_class=GNGTreeAgent, play_fun=play_grab_n_go, **{"env" : GrabNGoEnv(render_mode="non-human")})
        show_results(players, play_fun=play_grab_n_go, **{"env" : GrabNGoEnv(render_mode="non-human")}) #**{"env" : BoxingEnv(render_mode="non-human")})
    elif args.h:
        print(f"You can find the documentation at Evolutionary_Ranking_System/README.md")
    else:
        print("Invalid arguments")