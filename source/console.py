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
from chomp import *

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

if __name__ == '__main__':

    pass

    """rows = 10
    cols = 10
    poison_position = [-1, -1]
    players = [RandomIndividual(), RandomIndividual()]
    graphics = False

    result = play_chomp(rows=rows, cols=cols, poison_position=poison_position, players=players, graphics=graphics)
    print(result)"""