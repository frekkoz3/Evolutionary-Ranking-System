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

    try:
        env.render()
    except UserClosingWindowException as e:
        done, truncated = True, True

    while not done:    

        if isinstance(players[0], RandomIndividual) or isinstance(players[0], RealIndividual):
            action_a = players[0].move(env)
        else:
            action_a = players[0].move((obs, True)) # The boolean flag represent if the player is the first or the second
        if isinstance(players[1], RandomIndividual) or isinstance(players[1], RealIndividual):
            action_b = players[1].move(env)
        else:
            action_b = players[1].move((obs, False))
        
        obs, (r_a, r_b), done, truncated, info = env.step((action_a, action_b))

        # THIS COULD ALSO BEEN NON-IMPLEMENTED
        players[0].update(r_a)
        players[1].update(r_b)

        if done or truncated:
            if env.p1_score > env.p2_score:
                return (1, 0)
            return (0, 1)
        
        try:
            env.render()
        except UserClosingWindowException as e:
            done, truncated = True, True

    env.close()

if __name__ == '__main__':

    play_boxing(players=[RealIndividual(), RandomIndividual()], graphics=True)