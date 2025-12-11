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
    parser.add_argument("--help", type = bool, default=False, help="help message")

    # Parse
    args = parser.parse_args()

    if args.test:
        show_results(players=None, play_fun=play_grab_n_go, prefix="449_", **{"env" : GrabNGoEnv(render_mode="non-human")})
    elif args.train:
        ees = EES(config_path=os.path.join(CONFIG_DIR, "config.json"))
        #players = ees.play(player_class=DQNAgent, play_fun=play_boxing, **{"env" : BoxingEnv(render_mode="non-human")})
        players = ees.play(player_class=GNGTreeAgent, play_fun=play_grab_n_go, **{"env" : GrabNGoEnv(render_mode="non-human")})
        show_results(players, play_fun=play_grab_n_go, **{"env" : GrabNGoEnv(render_mode="non-human")}) #**{"env" : BoxingEnv(render_mode="non-human")})
    elif args.help:
        print(f"You can find the documentation at {os.path.join(PROJECT_ROOT, "README.md")}")
    else:
        print("Invalid arguments")