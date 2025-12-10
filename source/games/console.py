"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    console.py

    This file contain the handler for the virtual console to play various games.

    Function list:
        - play_boxing()
        - play_grab_n_go()
"""
from source.agents.individual import *
from source.agents.dqn_agent import *
from source.agents.grab_n_go_dqn_agent.gng_dqn_agent import *
from source.agents.grab_n_go_tree_agent.gng_tree_agent import *
from source.games.boxing.boxing import *
from source.games.grab_n_go.grab_n_go import *

def play_boxing(players = [RandomIndividual(), RandomIndividual()], render_mode = "human", eval_mode = True, **kwargs):
    """
        This play fun function follows the gym env protocol.
        This is good for RL bot.
    """
    env = BoxingEnv(render_mode=render_mode)

    obs, info = env.reset()
    obs_a = env.get_obs(perspective = 'p1')
    obs_b = env.get_obs(perspective = 'p2')

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
        elif isinstance(players[0], LogicalAIIndividual):
            action_a = players[0].move(env, 'p1')
        else:
            action_a = players[0].move(np.array(obs_a), env, eval_mode)
            
        if isinstance(players[1], RandomIndividual) or isinstance(players[1], RealIndividual):
            action_b = players[1].move(env)
        elif isinstance(players[1], LogicalAIIndividual):
            action_b = players[1].move(env, 'p2')
        else:
            action_b = players[1].move(np.array(obs_b), env, eval_mode)
        
        new_obs, (r_a, r_b), done, truncated, info = env.step((action_a, action_b))
        action_a = info['a1']
        action_b = info['a2']
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
                #print(f"player {players[0].id} WON against player {players[1].id}. Final score: {env.p1.score} - {env.p2.score}")
                return (1, 0)
            elif env.p2.score > env.p1.score:
                env.close()
                #print(f"player {players[1].id} WON against player {players[0].id}. Final score: {env.p1.score} - {env.p2.score}")
                return (0, 1)
            env.close()
            #print(f"player {players[1].id} and player {players[0].id} went EVEN. Final score: {env.p1.score} - {env.p2.score}")
            return (0, 0)
        
    env.close()

def play_grab_n_go(players = [RandomIndividual(), RandomIndividual()], render_mode = "human", eval_mode = True, **kwargs):
    """
        This play fun function follows the gym env protocol.
        This is good for RL bot.
    """
    env = GrabNGoEnv(render_mode=render_mode)

    obs, info = env.reset()
    obs_a = env.get_obs(perspective = 'p1', map = players[0].need_map())
    obs_b = env.get_obs(perspective = 'p2', map = players[1].need_map())

    done, truncated = False, False

    try:
        env.render()
    except UserClosingWindowException as e:
        done, truncated = True, True
        env.close()
        return (0, 0)

    while not done:    
        # catcher
        if isinstance(players[0], RandomIndividual) or isinstance(players[0], RealIndividual):
            action_a = players[0].move(env)
        elif isinstance(players[0], LogicalAIIndividual):
            action_a = players[0].move(env, 'p1')
        elif isinstance(players[0], GNGTreeAgent):
            action_a = players[0].move(obs = obs_a, **{"catcher" : True})
        else:
            action_a = players[0].move(state = np.array(obs_a), env = env, eval_mode = eval_mode, **{"catcher" : True})
        # runner
        if isinstance(players[1], RandomIndividual) or isinstance(players[1], RealIndividual):
            action_b = players[1].move(env)
        elif isinstance(players[1], LogicalAIIndividual):
            action_b = players[1].move(env, 'p2')
        elif isinstance(players[1], GNGTreeAgent):
            action_b = players[1].move(obs = obs_b, **{"catcher" : False})
        else:
            action_b = players[1].move(state = np.array(obs_b), env = env, eval_mode = eval_mode, **{"catcher" : False})
        
        new_obs, (r_a, r_b), done, truncated, info = env.step((action_a, action_b))
        """action_a = info['a1'] # it could be that it has been modified
        action_b = info['a2'] # it could be that it has been modified"""
        new_obs_a = env.get_obs('p1', map = players[0].need_map())
        new_obs_b = env.get_obs('p2', map = players[1].need_map())

        # OBSERVE THE ENVIRONMENT
        players[0].observe(np.array(obs_a).astype(np.float32) if not isinstance(obs_a, dict) else obs_a, action_a, r_a, np.array(new_obs_a).astype(np.float32) if not isinstance(obs_a, dict) else obs_a, done, **{"catcher" : True})
        players[1].observe(np.array(obs_b).astype(np.float32) if not isinstance(obs_b, dict) else obs_b, action_b, r_b, np.array(new_obs_b).astype(np.float32) if not isinstance(obs_b, dict) else obs_b, done, **{"catcher" : False})

        # UPDATE THE OBSERVATION
        obs_a = new_obs_a
        obs_b = new_obs_b

        # UPDATE THE INDIVIDUALS
        if not eval_mode:
            players[0].update(**{"catcher" : True})
            players[1].update(**{"catcher" : False})

        try:
            env.render()
        except UserClosingWindowException as e:
            done, truncated = True, True

        if done or truncated:
            if env.p1.score > env.p2.score:
                env.close()
                #print(f"player {players[0].id} WON against player {players[1].id}. Final score: {env.p1.score} - {env.p2.score}")
                return (1, 0)
            else:
                env.close()
                #print(f"player {players[1].id} WON against player {players[0].id}. Final score: {env.p1.score} - {env.p2.score}")
                return (0, 1)
        
    env.close()

if __name__ == '__main__':

    p1 = GNGTreeAgent(TreeAgent(100, 5), TreeAgent(100, 5))
    p2 = GNGTreeAgent(TreeAgent(100, 5), TreeAgent(100, 5))
    print(p1.catcher.trees_prob)
    print(p1.runner.trees_prob)
    print(p2.catcher.trees_prob)
    print(p2.runner.trees_prob)
    for i in range (100):
        res_1 = play_grab_n_go(players=[p1, p2], render_mode="non-human", eval_mode = False)
        res_2 = play_grab_n_go(players=[p2, p1], render_mode="non-human", eval_mode = False)
        print(res_1)
        print(res_2)
        p1.view_probs()
        p2.view_probs()
        print("-------------")
    play_grab_n_go(players=[p1, p2], render_mode="human", eval_mode = True)
    play_grab_n_go(players=[p2, p1], render_mode="human", eval_mode = True)