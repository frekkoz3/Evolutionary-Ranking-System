"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    console.py

    This file contain the handler for the virtual console to play various games.

    Function list:
        - play_boxing()
"""

from individual import *

from games.boxing.boxing import *

def play_boxing(players = [RandomIndividual(), RandomIndividual()], graphics = True, **kwargs):
    """
        This play fun function follows the gym env protocol.
        This is good for RL bot.
    """

    env = BoxingEnv(render_mode = "human" if graphics else "no")

    obs, info = env.reset()

    done = False

    while not done:    
        if isinstance(players[0], RandomIndividual):
            action_a = players[0].move(env)
        else:
            action_a = players[0].move((obs, True)) # The boolean flag represent if the player is the first or the second
        if isinstance(players[1], RandomIndividual):
            action_b = players[1].move(env)
        else:
            action_b = players[1].move((obs, False))
        
        obs, (r_a, r_b), done, truncated, info = env.step((action_a, action_b))

        try:
            env.render()
        except UserClosingWindowException as e:
            done, truncated = True, True

        # THIS COULD ALSO BEEN NON-IMPLEMENTED
        players[0].update(r_a)
        players[1].update(r_b)

        if done or truncated:
            if env.p1_score > env.p2_score:
                return (1, 0)
            return (0, 1)

    env.close()

if __name__ == '__main__':

    """
    CHOMP
    rows = 10 # number of rows on the chomp board
    cols = 10 # number of columns on the chomp board
    poison_position = [-1, -1] # position of the poisoned block on the chomp board. [-1, -1] = random
    
    kwargs = { "rows" : rows, "cols" : cols, "poison_position" : poison_position}

    play_chomp(graphics=True, players = [RandomIndividual(), RandomIndividual()], **kwargs)
    """

    """
    PINGPONG
    kwargs = { "width" : 800, "height" : 800, "paddle_height" : 100, "paddle_speed" : 6, "ball_speed" : 10, "speedup_factor": 1.05, "randomness" : 0.15}
    """

    #play_ping_pong(players=[PaddleTrackingIndividual(), PaddleTrackingIndividual()], **{ "width" : 800, "height" : 800, "paddle_height" : 100, "paddle_speed" : 6, "ball_speed" : 10, "speedup_factor": 1.05, "randomness" : 0.15})
    play_boxing()