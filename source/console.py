"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    console.py

    This file contain the handler for the virtual console to play various games.

    Function list:
        - play_boxing()
"""

from source.individual import *
from source.games.boxing.boxing import *

"""from individual import *
from games.boxing.boxing import *"""

def play_boxing(players = [RandomIndividual(), RandomIndividual()], render_mode = "human", eval_mode = True, **kwargs):
    """
        This play fun function follows the gym env protocol.
        This is good for RL bot.
    """
    env = BoxingEnv(render_mode=render_mode)

    obs, info = env.reset()
    obs_a = env.get_obs('p1')
    obs_b = env.get_obs('p2')

    done, truncated = False, False

    try:
        env.render()
    except UserClosingWindowException as e:
        done, truncated = True, True
        env.close()
        return (0, 0)

    while not done:    

        if isinstance(players[0], RandomIndividual) or isinstance(players[0], RealIndividual):
            action_a = players[0].move(env)
        else:
            action_a = players[0].move(np.array(obs_a), env) # The boolean flag represent if the player is the first or the second
        if isinstance(players[1], RandomIndividual) or isinstance(players[1], RealIndividual):
            action_b = players[1].move(env)
        else:
            action_b = players[1].move(np.array(obs_b), env)
        
        new_obs, (r_a, r_b), done, truncated, info = env.step((action_a, action_b))
        new_obs_a = env.get_obs('p1')
        new_obs_b = env.get_obs('p2')

        # OBSERVE THE ENVIRONMENT
        players[0].observe(np.array(obs_a).astype(np.float32), action_a, r_a, np.array(new_obs_a).astype(np.float32), done)
        players[1].observe(np.array(obs_b).astype(np.float32), action_b, r_b, np.array(new_obs_b).astype(np.float32), done)

        # UPDATE THE OBSERVATION
        obs_a = new_obs_a
        obs_b = new_obs_b

        # UPDATE THE INDIVIDUALS
        if not eval_mode:
            players[0].update()
            players[1].update()

        try:
            env.render()
        except UserClosingWindowException as e:
            done, truncated = True, True

        if done or truncated:
            if env.p1.score > env.p2.score:
                env.close()
                return (1, 0)
            elif env.p2.score > env.p1.score:
                env.close()
                return (0, 1)
            env.close()
            return (0, 0)
        
    env.close()

if __name__ == '__main__':

    play_boxing(players=[RealIndividual(), RandomIndividual()], render_mode="human")