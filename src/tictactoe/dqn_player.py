# © by Josua Bürki, 2019
# This is the class for the Deep Q-Learning Neural Network (DQN) Player for the game Tic-Tac-Toe
# An object of this class can have any name, but has to be different than the other objects
# The Neural Network is needed as input
# The reward_discount is the discount factor
# Furthermore the epsilon and the epsilon decay are needed for the Epsilon-Greedy-Policy
import random
import copy
from src import functions as f


class DQNPlayerTicTacToe:

    def __init__(self, name, nn, reward_discount=0.9, eps_start=1.0,
                 eps_end=0.01, eps_decay=0.99, update_target_network_every=10):
        self.name = name

        self.dnn = nn
        self.target_network = copy.deepcopy(self.dnn)
        self.update_every = update_target_network_every

        self.epsilon = 0.0
        self.epsilon_start = eps_start
        self.epsilon_end = eps_end
        self.epsilon_decay = eps_decay

        self.episode = None

        self.gamma = reward_discount

        self.experience = []

    @staticmethod
    def prepare_board(board):
        # Prepare the board for the Neural Network, make one list of floats
        board_state = []

        for i in range(len(board)):
            for j in range(len(board[0])):
                board_state.append([board[i][j]])
        return board_state

    def update_target_network(self, episode):
        if (episode % self.update_every) == 0:
            self.target_network = copy.deepcopy(self.dnn)

    def make_move(self, board, episode, training):
        # At first, the epsilon has to be determined
        # Afterwards the move can be chosen, just from the possible ones
        if training:
            if self.epsilon == 0.0:
                self.epsilon = self.epsilon_start
            if self.episode != episode:
                self.episode = episode
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        else:
            self.epsilon = 0.0

        q_values = self.dnn.query(self.prepare_board(board))
        [q_values] = f.matrix_transposition(q_values)

        for i in range(len(q_values)):
            if board[2][i] == 0.01:
                q_values[i] = -10

        if random.random() > self.epsilon:
            return q_values.index(max(q_values))
        else:
            while True:
                random_number = random.randint(0, 8)
                if q_values[random_number] != -10:
                    return random_number

    def buffer_experience(self, board, move, reward, board_next, turn):
        # Store the game step
        self.experience.append([self.prepare_board(board), move, reward, self.prepare_board(board_next), turn])

    def evolve(self, episode):
        self.update_target_network(episode)

        # Learn from the experience, it is different if it is the ending move or one before
        for i in reversed(range(len(self.experience))):
            if i < len(self.experience) - 2 and self.experience[i][4]:
                # Not ending moves
                target_q_values = self.dnn.query(self.experience[i][0])
                q_values_next = self.target_network.query(self.experience[i + 2][0])
                [max_q_value_next] = max(q_values_next)
                target_q_values[self.experience[i][1]] = [self.gamma * max_q_value_next]
                self.dnn.train(self.experience[i][0], target_q_values)

            elif i == len(self.experience) - 2 and self.experience[i][4]:
                # Lose or draw
                target_q_values = self.dnn.query(self.experience[i][0])
                target_q_values[self.experience[i][1]] = [self.experience[i + 1][2]]
                self.dnn.train(self.experience[i][0], target_q_values)

            elif i == len(self.experience) - 1 and self.experience[i][4]:
                # Win or draw
                target_q_values = self.dnn.query(self.experience[i][0])
                target_q_values[self.experience[i][1]] = [self.experience[i][2]]
                self.dnn.train(self.experience[i][0], target_q_values)

        self.experience = []
