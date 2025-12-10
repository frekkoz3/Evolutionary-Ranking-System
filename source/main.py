"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    main.py:
    
    This file is used just as final wrapper.
"""
from source.elo_system.elo_system import play
from source.games.console import play_grab_n_go
from source.agents.grab_n_go_tree_agent.gng_tree_agent import GNGTreeAgent
from source.agents.grab_n_go_dqn_agent.gng_dqn_agent import GNGDQNAgent
from source.games.grab_n_go.grab_n_go import GrabNGoEnv

if __name__ == '__main__':
    
    play(player_class=GNGTreeAgent, play_fun=play_grab_n_go, **{"env" : GrabNGoEnv(render_mode="non-human")})