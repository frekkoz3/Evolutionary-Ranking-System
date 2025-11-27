"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    console.py

    This file contain the handler for the virtual console to play various games.

    Function list:
        - play_boxing()
"""
import sys
import os

# Add the root of the project to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from source.individual import *

from source.games.boxing.boxing import *

def play_boxing(players = [RandomIndividual(), RandomIndividual()], render_mode = "human", **kwargs):
    """
        This play fun function follows the gym env protocol.
        This is good for RL bot.
    """

    env = BoxingEnv(render_mode)

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
            action_a = players[0].move(np.append(obs, True), env) # The boolean flag represent if the player is the first or the second
        if isinstance(players[1], RandomIndividual) or isinstance(players[1], RealIndividual):
            action_b = players[1].move(env)
        else:
            action_b = players[1].move(np.append(obs, False), env)
        
        new_obs, (r_a, r_b), done, truncated, info = env.step((action_a, action_b))

        extra_a = np.array([1.0])   # for player A
        extra_b = np.array([0.0])   # for player B

        # OBSERVE THE ENVIRONMENT
        players[0].observe(np.concatenate([obs, extra_a]).astype(np.float32), action_a, r_a, np.concatenate([new_obs, extra_a]).astype(np.float32), done)
        players[1].observe(np.concatenate([obs, extra_b]).astype(np.float32), action_b, r_b, np.concatenate([new_obs, extra_b]).astype(np.float32), done)

        # UPDATE THE OBSERVATION
        obs = new_obs

        # UPDATE THE INDIVIDUALS
        players[0].update()
        players[1].update()

        if done or truncated:
            if env.p1.score > env.p2.score:
                return (1, 0)
            elif env.p2.score > env.p1.score:
                return (0, 1)
            return (0, 0)
        
        try:
            env.render()
        except UserClosingWindowException as e:
            done, truncated = True, True

    env.close()

if __name__ == '__main__':

    play_boxing(players=[RealIndividual(), RandomIndividual()], graphics=True)