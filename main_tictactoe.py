from src import neural_network, q_table
from src.tictactoe import human_player, perfect_player, random_player, tic_tac_toe, dqn_player, tq_player

DNN = neural_network.DeepNeuralNetwork(0.01, number_of_hidden_layers=1, input_size=27, hidden_layer_size_1=100,
                                       output_size=9, initialise=True, saving_place="Weights_DNN.pkl")
QTable = q_table.QTable(19683, 9, initialize=True, content=0.6, saving_place="QTable.pkl")

# Deep Q-Learning Neural Network Players; the two train players save their epsilon, so use the test player for the tests
DQN_P1_train = dqn_player.DQNPlayerTicTacToe("DQN Player 1", DNN)
DQN_P2_train = dqn_player.DQNPlayerTicTacToe("DQN Player 2", DNN)
DQN_P_test = dqn_player.DQNPlayerTicTacToe("DQN Player Test", DNN)

# Tabular Q-Learning Players
TQ_P1_train = tq_player.TQPlayerTicTacToe("TQ Player 1", QTable)
TQ_P2_train = tq_player.TQPlayerTicTacToe("TQ Player 2", QTable)
TQ_P_test = tq_player.TQPlayerTicTacToe("TQ Player Test", QTable)

# Other Players
Random_P1 = random_player.RandomPlayerTicTacToe("Random")
Human_P1 = human_player.HumanPlayerTicTacToe("Human", "X")
Perfect_P1 = perfect_player.PerfectPlayerTicTacToe("Perfect")

# The game Tic-Tac-Toe
tictactoe = tic_tac_toe.TicTacToe()

# Test and train
tictactoe.play(DQN_P_test, Random_P1, episodes=1000)
for i in range(7):
    tictactoe.train(DQN_P1_train, Random_P1, episodes=10000)
    tictactoe.play(DQN_P_test, Random_P1, episodes=1000)
    QTable.save_table()

