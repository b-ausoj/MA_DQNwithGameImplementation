# © by Josua Bürki, 2019
# This is the class for the Human Player for the game Tic-Tac-Toe
# An object of this class can have any name, but has to be different than the other objects
# The game_stone has to be either "O" or "X"


class HumanPlayerTicTacToe:

    def __init__(self, name, game_stone):
        self.name = name

        self.board = None

        self.cross = "O"
        self.naught = "X"
        self.my_stone = game_stone

        if game_stone == self.cross:
            self.rival_stone = self.naught
        else:
            self.rival_stone = self.cross

    @staticmethod
    def draw_board(board):

        print('\n ' + board[0] + ' | ' + board[1] + ' | ' + board[2])
        print('-----------')
        print(' ' + board[3] + ' | ' + board[4] + ' | ' + board[5])
        print('-----------')
        print(' ' + board[6] + ' | ' + board[7] + ' | ' + board[8])

    def prepare_board(self, board):
        # Empty fields have their number and the occupied have the stones
        self.board = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

        for i in range(len(board[0])):
            if board[0][i] == 0.99:
                self.board[i] = self.my_stone
            elif board[1][i] == 0.99:
                self.board[i] = self.rival_stone

    def make_move(self, board, episode, training):
        # Episode and training are placeholders
        # They are needed for the function of other players

        self.prepare_board(board)
        self.draw_board(self.board)

        while True:
            players_choice = int(input("\nYour turn {} : ".format(self.name))) - 1
            if board[2][players_choice] == 0.99:
                return players_choice

    @staticmethod
    def buffer_experience(a, b, c, d, e):
        # Has no use, but is needed for the function of the training against this player
        pass

    @staticmethod
    def evolve():
        # Has no use, but is needed for the function of the training against this player
        pass
