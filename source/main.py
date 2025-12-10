"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    main.py:
    
    This file is used just as final wrapper.
"""

from source.games.console import *
from source.agents import *
from source.games import *

from source.elo_system.elo_system import EES, show_results

from source import CONFIG_DIR
import os

if __name__ == '__main__':
    
    ees = EES(config_path=os.path.join(CONFIG_DIR, "config.json"))
    #players = ees.play(player_class=DQNAgent, play_fun=play_boxing, **{"env" : BoxingEnv(render_mode="non-human")})
    players = ees.play(player_class=GNGTreeAgent, play_fun=play_grab_n_go, **{"env" : GrabNGoEnv(render_mode="non-human")})
    show_results(players, play_fun=play_grab_n_go, **{"env" : GrabNGoEnv(render_mode="non-human")}) #**{"env" : BoxingEnv(render_mode="non-human")})