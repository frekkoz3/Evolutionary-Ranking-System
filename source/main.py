"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    main.py:
    
    This file is used just as final wrapper.
"""
from elo_system import play
from console import play_boxing

if __name__ == '__main__':
    
    play(play_fun=play_boxing)