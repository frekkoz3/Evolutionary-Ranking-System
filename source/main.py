"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    main.py:
    
    This file is used just as final wrapper.
"""
from source.elo_system.elo_system import play
from source.games.console import play_boxing
from source.agents.dqn_agent.dqn_agent import DQNAgent
from source.games.boxing.boxing import BoxingEnv

if __name__ == '__main__':
    
    play(player_class=DQNAgent, play_fun=play_boxing, **{"env" : BoxingEnv(render_mode="non-human")})