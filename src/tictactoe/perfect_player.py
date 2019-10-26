# © by Josua Bürki, 2019
# This is the class for the Perfect Player for the game Tic-Tac-Toe
# An object of this class can have any name, but has to be different than the other objects
# This player plays after the strategy of Newell and Simon from the year 1972
from src import functions as f


class PerfectPlayerTicTacToe:

    def __init__(self, name):
        self.name = name

        self.win_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    def make_move(self, board, episode, training):
        # episode and training are placeholders
        # they are needed for the function of other players

        # 1. Win
        for a, b, c in self.win_combinations:
            if board[0][a] == board[0][b] == board[2][c] == 0.99:
                return c
            elif board[0][a] == board[0][c] == board[2][b] == 0.99:
                return b
            elif board[0][b] == board[0][c] == board[2][a] == 0.99:
                return a

        # 2. Block
        for a, b, c in self.win_combinations:
            if board[1][a] == board[1][b] == 0.99 and board[2][c] == 0.99:
                return c
            elif board[1][a] == board[1][c] == 0.99 and board[2][b] == 0.99:
                return b
            elif board[1][b] == board[1][c] == 0.99 and board[2][a] == 0.99:
                return a

        # 3. Fork
        for a, b, c in self.win_combinations:
            for d, e, f in self.win_combinations:
                if a == (d or e or f) or b == (d or e or f) or c == (d or e or f):
                    if board[2][a] == board[2][b] == board[0][c] == 0.99:
                        if board[2][d] == board[2][e] == board[0][f] == 0.99:
                            if c != f:
                                if a == (d or e):
                                    return a
                                elif b == (d or e):
                                    return b
                        elif board[2][d] == board[2][f] == board[0][e] == 0.99:
                            if c != e:
                                if a == (d or f):
                                    return a
                                elif b == (d or f):
                                    return b
                        elif board[2][e] == board[2][f] == board[0][d] == 0.99:
                            if c != d:
                                if a == (e or f):
                                    return a
                                elif b == (e or f):
                                    return b
                    elif board[2][a] == board[2][c] == board[0][b] == 0.99:
                        if board[2][d] == board[2][e] == board[0][f] == 0.99:
                            if b != f:
                                if a == (d or e):
                                    return a
                                elif c == (d or e):
                                    return c
                        elif board[2][d] == board[2][f] == board[0][e] == 0.99:
                            if b != e:
                                if a == (d or f):
                                    return a
                                elif c == (d or f):
                                    return c
                        elif board[2][e] == board[2][f] == board[0][d] == 0.99:
                            if b != d:
                                if a == (e or f):
                                    return a
                                elif c == (e or f):
                                    return c
                    elif board[2][b] == board[2][c] == board[0][a] == 0.99:
                        if board[2][d] == board[2][e] == board[0][f] == 0.99:
                            if a != f:
                                if b == (d or e):
                                    return b
                                elif c == (d or e):
                                    return c
                        elif board[2][d] == board[2][f] == board[0][e] == 0.99:
                            if a != e:
                                if b == (d or f):
                                    return b
                                elif c == (d or f):
                                    return c
                        elif board[2][e] == board[2][f] == board[0][d] == 0.99:
                            if a != d:
                                if b == (e or f):
                                    return b
                                elif c == (e or f):
                                    return c

        # 4. Block Fork
        for a, b, c in self.win_combinations:
            for d, e, f in self.win_combinations:
                if board[2][a] == board[2][b] == board[1][c] == 0.99:
                    if board[2][d] == board[2][e] == board[1][f] == 0.99:
                        if (c != f) and (a == d or e):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != a:
                                        return self.check_move(board, h, g)
                                    elif h != a:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != a:
                                        return self.check_move(board, i, g)
                                    elif i != a:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != a:
                                        return self.check_move(board, h, i)
                                    elif h != a:
                                        return self.check_move(board, i, h)
                            return a
                        elif (c != f) and (b == d or e):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != b:
                                        return self.check_move(board, h, g)
                                    elif h != b:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != b:
                                        return self.check_move(board, i, g)
                                    elif i != b:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != b:
                                        return self.check_move(board, h, i)
                                    elif h != b:
                                        return self.check_move(board, i, h)
                            return b
                    elif board[2][d] == board[2][f] == board[1][e] == 0.99:
                        if (c != e) and (a == d or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != a:
                                        return self.check_move(board, h, g)
                                    elif h != a:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != a:
                                        return self.check_move(board, i, g)
                                    elif i != a:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != a:
                                        return self.check_move(board, h, i)
                                    elif h != a:
                                        return self.check_move(board, i, h)
                            return a
                        elif (c != e) and (b == d or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != b:
                                        return self.check_move(board, h, g)
                                    elif h != b:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != b:
                                        return self.check_move(board, i, g)
                                    elif i != b:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != b:
                                        return self.check_move(board, h, i)
                                    elif h != b:
                                        return self.check_move(board, i, h)
                            return b
                    elif board[2][e] == board[2][f] == board[1][d] == 0.99:
                        if (c != d) and (a == e or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != a:
                                        return self.check_move(board, h, g)
                                    elif h != a:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != a:
                                        return self.check_move(board, i, g)
                                    elif i != a:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != a:
                                        return self.check_move(board, h, i)
                                    elif h != a:
                                        return self.check_move(board, i, h)
                            return a
                        elif (c != d) and (b == e or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != b:
                                        return self.check_move(board, h, g)
                                    elif h != b:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != b:
                                        return self.check_move(board, i, g)
                                    elif i != b:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != b:
                                        return self.check_move(board, h, i)
                                    elif h != b:
                                        return self.check_move(board, i, h)
                            return b
                elif board[2][a] == board[2][c] == board[1][b] == 0.99:
                    if board[2][d] == board[2][e] == board[1][f] == 0.99:
                        if (b != f) and (a == d or e):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != a:
                                        return self.check_move(board, h, g)
                                    elif h != a:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != a:
                                        return self.check_move(board, i, g)
                                    elif i != a:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != a:
                                        return self.check_move(board, h, i)
                                    elif h != a:
                                        return self.check_move(board, i, h)
                            return a
                        elif (b != f) and (c == d or e):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != c:
                                        return self.check_move(board, h, g)
                                    elif h != c:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != c:
                                        return self.check_move(board, i, g)
                                    elif i != c:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != c:
                                        return self.check_move(board, h, i)
                                    elif h != c:
                                        return self.check_move(board, i, h)
                            return c
                    elif board[2][d] == board[2][f] == board[1][e] == 0.99:
                        if (b != e) and (a == d or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != a:
                                        return self.check_move(board, h, g)
                                    elif h != a:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != a:
                                        return self.check_move(board, i, g)
                                    elif i != a:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != a:
                                        return self.check_move(board, h, i)
                                    elif h != a:
                                        return self.check_move(board, i, h)
                            return a
                        elif (b != e) and (c == d or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != c:
                                        return self.check_move(board, h, g)
                                    elif h != c:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != c:
                                        return self.check_move(board, i, g)
                                    elif i != c:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != c:
                                        return self.check_move(board, h, i)
                                    elif h != c:
                                        return self.check_move(board, i, h)
                            return c
                    elif board[2][e] == board[2][f] == board[1][d] == 0.99:
                        if (b != d) and (a == e or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != a:
                                        return self.check_move(board, h, g)
                                    elif h != a:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != a:
                                        return self.check_move(board, i, g)
                                    elif i != a:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != a:
                                        return self.check_move(board, h, i)
                                    elif h != a:
                                        return self.check_move(board, i, h)
                            return a
                        elif (b != d) and (c == e or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != c:
                                        return self.check_move(board, h, g)
                                    elif h != c:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != c:
                                        return self.check_move(board, i, g)
                                    elif i != c:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != c:
                                        return self.check_move(board, h, i)
                                    elif h != c:
                                        return self.check_move(board, i, h)
                            return c
                elif board[2][b] == board[2][c] == board[1][a] == 0.99:
                    if board[2][d] == board[2][e] == board[1][f] == 0.99:
                        if (a != f) and (b == d or e):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != b:
                                        return self.check_move(board, h, g)
                                    elif h != b:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != b:
                                        return self.check_move(board, i, g)
                                    elif i != b:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != b:
                                        return self.check_move(board, h, i)
                                    elif h != b:
                                        return self.check_move(board, i, h)
                            return b
                        elif (a != f) and (c == d or e):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != c:
                                        return self.check_move(board, h, g)
                                    elif h != c:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != c:
                                        return self.check_move(board, i, g)
                                    elif i != c:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != c:
                                        return self.check_move(board, h, i)
                                    elif h != c:
                                        return self.check_move(board, i, h)
                            return c
                    if board[2][d] == board[2][f] == board[1][e] == 0.99:
                        if (a != e) and (b == d or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != b:
                                        return self.check_move(board, h, g)
                                    elif h != b:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != b:
                                        return self.check_move(board, i, g)
                                    elif i != b:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != b:
                                        return self.check_move(board, h, i)
                                    elif h != b:
                                        return self.check_move(board, i, h)
                            return b
                        elif (a != e) and (c == d or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != c:
                                        return self.check_move(board, h, g)
                                    elif h != c:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != c:
                                        return self.check_move(board, i, g)
                                    elif i != c:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != c:
                                        return self.check_move(board, h, i)
                                    elif h != c:
                                        return self.check_move(board, i, h)
                            return c
                    if board[2][e] == board[2][f] == board[1][d] == 0.99:
                        if (a != d) and (b == e or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != b:
                                        return self.check_move(board, h, g)
                                    elif h != b:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != b:
                                        return self.check_move(board, i, g)
                                    elif i != b:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != b:
                                        return self.check_move(board, h, i)
                                    elif h != b:
                                        return self.check_move(board, i, h)
                            return b
                        elif (a != d) and (c == e or f):
                            for g, h, i in self.win_combinations:
                                if board[2][g] == board[2][h] == board[0][i] == 0.99:
                                    if g != c:
                                        return self.check_move(board, h, g)
                                    elif h != c:
                                        return self.check_move(board, g, h)
                                elif board[2][g] == board[2][i] == board[0][h] == 0.99:
                                    if g != c:
                                        return self.check_move(board, i, g)
                                    elif i != c:
                                        return self.check_move(board, g, i)
                                elif board[2][h] == board[2][i] == board[0][g] == 0.99:
                                    if i != c:
                                        return self.check_move(board, h, i)
                                    elif h != c:
                                        return self.check_move(board, i, h)
                            return c

        # 5. Play Center
        if board[2][4] == 0.99:
            return 4

        # 6. Play Opposite Corner
        if board[1][0] == 0.99:
            if board[2][8] == 0.99:
                return 8
        elif board[1][2] == 0.99:
            if board[2][6] == 0.99:
                return 6
        elif board[1][6] == 0.99:
            if board[2][2] == 0.99:
                return 2
        elif board[1][8] == 0.99:
            if board[2][0] == 0.99:
                return 0

        # 7. Play Empty Corner
        if board[2][0] == 0.99:
            return 0
        elif board[2][2] == 0.99:
            return 2
        elif board[2][6] == 0.99:
            return 6
        elif board[2][8] == 0.99:
            return 8

        # 8. Play Empty Side
        if board[2][1] == 0.99:
            return 1
        elif board[2][3] == 0.99:
            return 3
        elif board[2][5] == 0.99:
            return 5
        elif board[2][7] == 0.99:
            return 7

    def check_move(self, board, x, y):

        board_test = f.copy_board(board)

        board_test[1][y] = 0.99
        board_test[2][y] = 0.01

        for a, b, c in self.win_combinations:
            if board_test[1][a] == board_test[1][b] == 0.99 and board_test[2][c] == 0.99:
                return y
            elif board_test[1][a] == board_test[1][c] == 0.99 and board_test[2][b] == 0.99:
                return y
            elif board_test[1][b] == board_test[1][c] == 0.99 and board_test[2][a] == 0.99:
                return y

        return x

    @staticmethod
    def buffer_experience(a, b, c, d, e):
        # Has no use, but is needed for the function of the training against this player
        pass

    @staticmethod
    def evolve():
        # Has no use, but is needed for the function of the training against this player
        pass
