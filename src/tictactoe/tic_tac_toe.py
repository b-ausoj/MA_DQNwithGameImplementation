# © by Josua Bürki, 2019
# This is the class for the game Tic-Tac-Toe
# The reward is needed by the initialization so the players know if they won or lost
# Random for who goes first
import random
from src import functions as f


class TicTacToe:

    def __init__(self, win_reward=1.0, lose_reward=0.0, draw_reward=0.5):

        self.win_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

        self.player_x_turn = None
        self.not_player_x_turn = None
        self.name_player_1 = None
        self.name_player_2 = None

        self.board = None
        self.board_next = None
        self.board_empty = [[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                            [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]]

        self.move = None

        self.win_value = win_reward
        self.lose_value = lose_reward
        self.draw_value = draw_reward
        self.reward = None
        self.reward_opponent = None

        self.game_playing = True

        self.result = None
        self.statistics = None
        self.statistics_total = None
        self.statistic_empty = None

    def set_up_game(self, player_1, player_2):
        # Before every game, to set the variables to 0 and chose who goes first

        self.board = f.copy_board(self.board_empty)
        self.reward = 0
        self.reward_opponent = 0

        if random.randint(0, 1) == 0:
            self.player_x_turn, self.not_player_x_turn = player_1, player_2
        else:
            self.player_x_turn, self.not_player_x_turn = player_2, player_1

        self.game_playing = True

    def update_board(self):
        # Update the board with the move and check if the game ends

        self.board_next = f.copy_board(self.board)

        self.board_next[0][self.move] = 0.99
        self.board_next[2][self.move] = 0.01

        self.check_board()

    def check_board(self):
        # Check for a winner and a tie

        for a, b, c in self.win_combinations:
            if self.board_next[0][a] == self.board_next[0][b] == self.board_next[0][c] == 0.99:
                self.reward = self.win_value
                self.reward_opponent = self.lose_value

                self.game_playing = False
                self.result = self.player_x_turn
                return

            elif self.board_next[2].count(0.99) == 0:
                self.reward = self.draw_value
                self.reward_opponent = self.draw_value

                self.game_playing = False
                self.result = True
                return

    def change_board(self):
        # After every turn has the board to be changed for the other player
        # This is because for both player the same stone is their
        self.board = f.copy_board(self.board_next)

        for i in range(len(self.board[0])):
            if self.board[0][i] == 0.01 and self.board[1][i] == 0.99:
                self.board[0][i] = 0.99
                self.board[1][i] = 0.01
            elif self.board[0][i] == 0.99 and self.board[1][i] == 0.01:
                self.board[0][i] = 0.01
                self.board[1][i] = 0.99

    def end_of_turn(self, player_1, player_2):
        # Change whose turn it is
        self.change_board()

        if self.player_x_turn == player_1:
            self.player_x_turn, self.not_player_x_turn = player_2, player_1
        else:
            self.player_x_turn, self.not_player_x_turn = player_1, player_2

    def update_statistics(self, episode, player_1, player_2):

        if self.result == player_1:
            self.statistics[self.name_player_1] += 1
            self.statistics_total[self.name_player_1] += 1
        elif self.result == player_2:
            self.statistics[self.name_player_2] += 1
            self.statistics_total[self.name_player_2] += 1
        elif self.result:
            self.statistics["Draw"] += 1
            self.statistics_total["Draw"] += 1

        if (episode + 1) % 1000 == 0:
            print("After episode", episode + 1, ": ", self.statistics)
            # To make graphs:
            # print(episode + 1, self.statistics[self.name_player_1], self.statistics[self.name_player_2], self.statistics["Draw"])
            self.statistics = self.statistic_empty.copy()

    def train(self, player1, player2, episodes):
        # The function to train
        self.name_player_1 = "{}".format(player1.name)
        self.name_player_2 = "{}".format(player2.name)

        self.statistic_empty = {self.name_player_1: 0, self.name_player_2: 0, "Draw": 0}
        self.statistics, self.statistics_total = self.statistic_empty.copy(), self.statistic_empty.copy()

        for episode in range(episodes):
            self.set_up_game(player1, player2)
            while self.game_playing:
                self.move = self.player_x_turn.make_move(self.board, episode, True)
                self.update_board()

                self.player_x_turn.buffer_experience(self.board, self.move, self.reward, self.board_next, True)
                self.not_player_x_turn.buffer_experience(self.board, self.move, self.reward_opponent, self.board_next,
                                                         False)
                self.end_of_turn(player1, player2)

            self.update_statistics(episode, player1, player2)

            player1.evolve()
            player2.evolve()

        print("Final statistics: ", self.statistics_total, "\n")

    def play(self, player1, player2, episodes):
        # The function to play and test, similar to train but a bit faster
        self.name_player_1 = "{}".format(player1.name)
        self.name_player_2 = "{}".format(player2.name)

        self.statistic_empty = {self.name_player_1: 0, self.name_player_2: 0, "Draw": 0}
        self.statistics, self.statistics_total = self.statistic_empty.copy(), self.statistic_empty.copy()

        for episode in range(episodes):
            self.set_up_game(player1, player2)
            while self.game_playing:
                self.move = self.player_x_turn.make_move(self.board, episode, False)
                self.update_board()

                self.end_of_turn(player1, player2)

            self.update_statistics(episode, player1, player2)

        print("Final statistics: ", self.statistics_total, "\n")
