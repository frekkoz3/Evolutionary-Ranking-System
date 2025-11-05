"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco
"""
import random
import copy

class Chomp:
    def __init__(self, rows=4, cols=6, poison_position : list = [-1, -1]):
        """
            Poison position can be passed as parameter or just as (-1, -1) to sample it uniformly
        """
        self.rows = rows
        self.cols = cols
        self.board = [[1 for _ in range(cols)] for _ in range(rows)]
        self.game_over = False
        self.poison_position = [random.randrange(0, rows), random.randrange(0, cols)] if tuple(poison_position) == (-1, -1) else poison_position

    def display(self):
        print("\nBoard:")
        col_str = "  "
        for c in range(self.cols):
            col_str += f"{c}  "
        print(col_str)
        for r in range(self.rows):
            row_str = f"{r} "
            for c in range(self.cols):
                if r == self.poison_position[0] and c == self.poison_position[1]:
                    row_str += "💀 "
                elif self.board[r][c] == 1:
                    row_str += "🍫 "
                else:
                    row_str += "⬛ "
                
            print(row_str)
        print()

    def valid_moves(self):
        moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == 1:
                    moves.append((r, c))
        return moves

    def apply_move(self, move):
        r, c = move
        for i in range(r, self.rows):
            for j in range(c, self.cols):
                self.board[i][j] = 0

        # Check if the player ate the poison (bottom-left)
        if self.board[self.poison_position[0]][self.poison_position[1]] == 0:
            self.game_over = True

    def copy(self):
        clone = Chomp(self.rows, self.cols)
        clone.board = copy.deepcopy(self.board)
        clone.game_over = self.game_over
        return clone
    
    def get_state(self):
        return self.board
    
if __name__ == "__main__":
    pass
