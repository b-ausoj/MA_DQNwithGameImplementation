# © by Josua Bürki, 2019
# This is the class for the Human Player for the game Tic-Tac-Toe
# An object of this class can have any name, but has to be different than the other objects
import random


class RandomPlayerTicTacToe:

    def __init__(self, name):
        self.name = name

    @staticmethod
    def make_move(board, episode, training):
        # episode and training are placeholders
        # they are needed for the function of other players

        while True:
            random_number = random.randint(0, 8)
            if board[2][random_number] == 0.99:
                return random_number

    @staticmethod
    def buffer_experience(a, b, c, d, e):
        # Has no use, but is needed for the function of the training against this player
        pass

    @staticmethod
    def evolve(e):
        # Has no use, but is needed for the function of the training against this player
        pass
