# © by Josua Bürki, 2019
# This is the class for the Neural Network
import pickle
from src import functions as f


class DeepNeuralNetwork:

    def __init__(self, learning_rate, number_of_hidden_layers, input_size=27, hidden_layer_size_1=9,
                 hidden_layer_size_2=9, output_size=9, saving_place="Weights_DNN.pkl", initialise=True, weights=True):

        # Number of Hidden Layers: 0 - 2
        self.number_hidden_layers = number_of_hidden_layers
        self.input_nodes = input_size
        self.hidden_nodes_1 = hidden_layer_size_1
        self.hidden_nodes_2 = hidden_layer_size_2
        self.output_nodes = output_size

        self.save = saving_place

        if self.number_hidden_layers == 0:
            if initialise:
                if weights is True:
                    self.wio = f.random_numbers_gauss(0.0, pow(self.output_nodes, -0.5), self.output_nodes,
                                                      self.input_nodes)
                else:
                    self.wio = f.create_matrix(self.output_nodes, self.input_nodes, weights)
            else:
                with open(self.save, "rb") as read_weights:
                    self.wio = pickle.load(read_weights)

        elif self.number_hidden_layers == 1:
            if initialise:
                if weights is True:
                    self.wih = f.random_numbers_gauss(0.0, pow(self.hidden_nodes_1, -0.5), self.hidden_nodes_1,
                                                      self.input_nodes)
                    self.who = f.random_numbers_gauss(0.0, pow(self.output_nodes, -0.5), self.output_nodes,
                                                      self.hidden_nodes_1)
                else:
                    self.wih = f.create_matrix(self.hidden_nodes_1, self.input_nodes, weights)
                    self.who = f.create_matrix(self.output_nodes, self.hidden_nodes_1, weights)
            else:
                with open(self.save, "rb") as read_weights:
                    self.wih, self.who = pickle.load(read_weights)

        elif self.number_hidden_layers == 2:
            if initialise:
                if weights is True:
                    self.wih1 = f.random_numbers_gauss(0.0, pow(self.hidden_nodes_1, -0.5), self.hidden_nodes_1,
                                                            self.input_nodes)
                    self.wh1h2 = f.random_numbers_gauss(0.0, pow(self.hidden_nodes_2, -0.5), self.hidden_nodes_2,
                                                            self.hidden_nodes_1)
                    self.wh2o = f.random_numbers_gauss(0.0, pow(self.output_nodes, -0.5), self.output_nodes,
                                                            self.hidden_nodes_2)
                else:
                    self.wih1 = f.create_matrix(self.hidden_nodes_1, self.input_nodes, weights)
                    self.wh1h2 = f.create_matrix(self.hidden_nodes_2, self.hidden_nodes_1, weights)
                    self.wh2o = f.create_matrix(self.output_nodes, self.hidden_nodes_2, weights)
            else:
                with open(self.save, "rb") as read_weights:
                    self.wih1, self.wh1h2, self.wh2o = pickle.load(read_weights)
        else:
            print("Unavailable number of hidden layers")

        self.lr = learning_rate
        self.activation_function = lambda x: f.leaky_relu(x)
        self.derivation = lambda x: f.leaky_relu_derivation(x)

    def save_weights(self):
        if self.number_hidden_layers == 0:
            with open(self.save, "bw") as write_weights:
                weights = self.wio
                pickle.dump(weights, write_weights)
        elif self.number_hidden_layers == 1:
            with open(self.save, "bw") as write_weights:
                weights = self.wih, self.who
                pickle.dump(weights, write_weights)
        elif self.number_hidden_layers == 2:
            with open(self.save, "bw") as write_weights:
                weights = self.wih1, self.wh1h2, self.wh2o
                pickle.dump(weights, write_weights)

    def query(self, input_state):
        if self.number_hidden_layers == 0:
            inputs = f.copy_board(input_state)

            final_inputs = f.matrix_multiplication(self.wio, inputs)
            final_outputs = self.activation_function(final_inputs)

            return final_outputs
        elif self.number_hidden_layers == 1:
            inputs = f.copy_board(input_state)

            hidden_inputs = f.matrix_multiplication(self.wih, inputs)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = f.matrix_multiplication(self.who, hidden_outputs)
            final_outputs = self.activation_function(final_inputs)

            return final_outputs
        elif self.number_hidden_layers == 2:
            inputs = f.copy_board(input_state)

            hidden_1_inputs = f.matrix_multiplication(self.wih1, inputs)
            hidden_1_outputs = self.activation_function(hidden_1_inputs)

            hidden_2_inputs = f.matrix_multiplication(self.wh1h2, hidden_1_outputs)
            hidden_2_outputs = self.activation_function(hidden_2_inputs)

            final_inputs = f.matrix_multiplication(self.wh2o, hidden_2_outputs)
            final_outputs = self.activation_function(final_inputs)

            return final_outputs

    def train(self, input_state, targets):
        # Training for NN with 0 hidden layers
        if self.number_hidden_layers == 0:
            inputs = f.copy_board(input_state)

            final_inputs = f.matrix_multiplication(self.wio, inputs)
            final_outputs = self.activation_function(final_inputs)

            output_errors = f.subtraction(targets, final_outputs)

            self.wio = f.addition(self.wio, f.multiplication(self.lr, f.matrix_multiplication(f.multiplication(
                output_errors, self.derivation(final_outputs)), f.matrix_transposition(inputs))))

        elif self.number_hidden_layers == 1:
            # Training for NN with 1 hidden layers
            inputs = f.copy_board(input_state)

            hidden_inputs = f.matrix_multiplication(self.wih, inputs)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = f.matrix_multiplication(self.who, hidden_outputs)
            final_outputs = self.activation_function(final_inputs)

            output_errors = f.subtraction(targets, final_outputs)

            hidden_errors = f.matrix_multiplication(f.matrix_transposition(self.who), output_errors)

            self.who = f.addition(self.who, f.multiplication(self.lr, f.matrix_multiplication(f.multiplication(
                output_errors, self.derivation(final_outputs)), f.matrix_transposition(hidden_outputs))))
            self.wih = f.addition(self.wih, f.multiplication(self.lr, f.matrix_multiplication(f.multiplication(
                hidden_errors, self.derivation(hidden_outputs)), f.matrix_transposition(inputs))))

        elif self.number_hidden_layers == 2:
            # Training for NN with 2 hidden layers
            inputs = f.copy_board(input_state)

            hidden_1_inputs = f.matrix_multiplication(self.wih1, inputs)
            hidden_1_outputs = self.activation_function(hidden_1_inputs)

            hidden_2_inputs = f.matrix_multiplication(self.wh1h2, hidden_1_outputs)
            hidden_2_outputs = self.activation_function(hidden_2_inputs)

            final_inputs = f.matrix_multiplication(self.wh2o, hidden_2_outputs)
            final_outputs = self.activation_function(final_inputs)

            output_errors = f.subtraction(targets, final_outputs)

            hidden_errors_2 = f.matrix_multiplication(f.matrix_transposition(self.wh2o), output_errors)
            hidden_errors_1 = f.matrix_multiplication(f.matrix_transposition(self.wh1h2), hidden_errors_2)

            self.wh2o = f.addition(self.wh2o, f.multiplication(self.lr, f.matrix_multiplication(f.multiplication(
                output_errors, self.derivation(final_outputs)), f.matrix_transposition(hidden_2_outputs))))
            self.wh1h2 = f.addition(self.wh1h2, f.multiplication(self.lr, f.matrix_multiplication(f.multiplication(
                hidden_errors_2, self.derivation(hidden_2_outputs)), f.matrix_transposition(hidden_1_outputs))))
            self.wih1 = f.addition(self.wih1, f.multiplication(self.lr, f.matrix_multiplication(f.multiplication(
                hidden_errors_1, self.derivation(hidden_1_outputs)), f.matrix_transposition(inputs))))
