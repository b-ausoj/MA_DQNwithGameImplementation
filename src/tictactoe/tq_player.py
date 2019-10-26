# © by Josua Bürki, 2019
# This is the class for the Tabular Q-Learning (TQ) Player for the game Tic-Tac-Toe
# An object of this class can have any name, but has to be different than the other objects
# The Q Table is needed as input
# The reward_discount is the discount factor and alpha the learning rate
# Furthermore the epsilon and the epsilon decay are not needed, faster without
from src import functions as f


class TQPlayerTicTacToe:

    def __init__(self, name, q_table, learning_rate=0.1, reward_discount=0.9):
        self.name = name

        self.q = q_table

        self.episode = None

        self.alpha = learning_rate
        self.gamma = reward_discount

        self.experience = []

    @staticmethod
    def prepare_board(board):
        # Prepare the board for the Q Table, make one list with 9 elements (0, 1 or 2)
        board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(len(board_state)):
            if board[0][i] == 0.99:
                board_state[i] = 2
        for j in range(len(board_state)):
            if board[1][j] == 0.99:
                board_state[j] = 1
        return board_state

    def make_move(self, board, episode, training):
        # The move is the highest values from the possible moves
        q_values = self.q.table[f.board_to_number(self.prepare_board(board))]
        for i in range(len(q_values)):
            if board[2][i] == 0.01:
                q_values[i] = -10

        return q_values.index(max(q_values))

    def buffer_experience(self, board, move, reward, board_next, turn):
        # Store the game step
        self.experience.append([self.prepare_board(board), move, reward, self.prepare_board(board_next), turn])

    def evolve(self):
        # Learn from the experience, it is different if it is the ending move or one before
        for i in reversed(range(len(self.experience))):
            if i < len(self.experience) - 2 and self.experience[i][4]:
                # Not ending moves
                board = f.board_to_number(self.experience[i][0])
                board_next = f.board_to_number(self.experience[i + 2][0])
                self.q.table[board][self.experience[i][1]] = (1.0 - self.alpha) * self.q.table[board][
                    self.experience[i][1]] + self.alpha * self.gamma * max(
                        self.q.table[board_next])

            elif i == len(self.experience) - 2 and self.experience[i][4]:
                # Lose or draw
                board = f.board_to_number(self.experience[i][0])
                self.q.table[board][self.experience[i][1]] = (1.0 - self.alpha) * self.q.table[board][
                    self.experience[i][1]] + self.alpha * self.experience[i + 1][2]

            elif i == len(self.experience) - 1 and self.experience[i][4]:
                # Win or draw
                board = f.board_to_number(self.experience[i][0])
                self.q.table[board][self.experience[i][1]] = (1.0 - self.alpha) * self.q.table[board][
                    self.experience[i][1]] + self.alpha * self.experience[i][2]

        self.experience = []
