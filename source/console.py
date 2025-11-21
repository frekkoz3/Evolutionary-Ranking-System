"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    chomp.py

    This file contain the handler for the virtual console to play the game "Chomp".

    Function list:
        - show_end_screen(turn)
        - play_chomp(rows, cols, poison_position, players, graphics)
"""

from individual import *
from games.chomp.chomp import *

from games.ping_pong.ping_pong import *

def show_end_screen(turn):
    print("\n" + "="*50)
    print(f"💀  Player {turn} loses 💀".center(50))
    print("="*50)

def play_chomp(players = [RealIndividual(), RandomIndividual()], graphics = True, **kwargs):

    rows = kwargs["rows"]
    cols = kwargs["cols"]
    poison_position = kwargs["poison_position"]

    game = Chomp(rows=rows, cols=cols, poison_position=poison_position)
    turn = random.randint(0, 1) 
    result = (0, 0)

    if graphics:
        print(f"\n🍫 Welcome to Chomp ({rows}x{cols})!")
        print(f"The piece at position ({game.poison_position}) is poisoned. Don't eat it!\n")

    while not game.game_over:
        if graphics:
            game.display()

        move = players[turn].move(game)
        game.apply_move(move)
        if graphics:
            print(f"Player {turn}'s move : {move}")
        if game.game_over:
            result = (1, 0) if turn == 1 else (0, 1)
            if graphics:
                show_end_screen(turn)
            return result

        turn = turn + 1
        turn = turn % 2

def play_ping_pong(players = [RandomIndividual(), RandomIndividual()], graphics = True, **kwargs):
    """
        This play fun function follows the gym env protocol.
        This is good for RL bot.
    """

    env = PongEnv(**kwargs)
    env.reset(done = True)

    state = env._get_state()

    while True:

        if isinstance(players[0], RandomIndividual):
            action_a = players[0].move(env)
        else:
            action_a = players[0].move((state, True)) # The boolean flag represent if the player is the first or the second
        if isinstance(players[1], RandomIndividual):
            action_b = players[1].move(env)
        else:
            action_b = players[1].move((state, False))

        state, (r_a, r_b), done, scored, _ = env.step(action_a, action_b)

        # THIS COULD ALSO BEEN NON-IMPLEMENTED
        players[0].update(r_a)
        players[1].update(r_b)

        if graphics:
            env.render_ascii()
            time.sleep(0.01)
        
        if done:
            if env.score_a > env.score_b:
                return (1, 0)
            return (0, 1)

        if scored:
            env.reset(done)

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

    play_ping_pong(players=[PaddleTrackingIndividual(), PaddleTrackingIndividual()], **{ "width" : 800, "height" : 800, "paddle_height" : 100, "paddle_speed" : 6, "ball_speed" : 10, "speedup_factor": 1.05, "randomness" : 0.15})
    