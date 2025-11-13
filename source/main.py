"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    main.py:
    
    This file is used just as final wrapper.
"""
from game import play
from individual import PaddleTrackingIndividual
from console import play_ping_pong

if __name__ == '__main__':
    
    play(player_class=PaddleTrackingIndividual, play_fun=play_ping_pong, **{ "width" : 800, "height" : 800, "paddle_height" : 100, "paddle_speed" : 6, "ball_speed" : 10, "speedup_factor": 1.05, "randomness" : 0.15})