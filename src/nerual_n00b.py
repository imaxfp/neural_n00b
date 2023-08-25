import unittest
from typing import List
from collections import deque
import numpy as np
import math
from matrix_n00b import MatrixNoob
from matrix_n00b_np import MatrixNoobNp
import pickle


""" 
The main idea of the Neural Netsork is Universal Approximators:

The process of finding an approximation or estimate of a mathematical function based on a set of input and output data points.
The goal is to find a function that can accurately predict the output value for any given input.

How powerful a deep network could be. 
This question has been answered multiple times, e.g. George Cybenko (1989) with roots from Ukraine, 
published his work that a feed-forward artificial neural network (in which connections do not form cycles) with one hidden layer
can approximate any continuous function of multiple variables with any desired accuracy.

Micchelli (1984) in the context of reproducing kernel Hilbert spaces in a way that could be seen as radial basis function (RBF) 
networks with a single hidden layer. These (and related results) suggest that even with a single-hidden-layer network, 
given enough nodes (possibly absurdly many), and the right set of weights, we can model any function.

Moreover, just because a single-hidden-layer network can learn any function does not mean that you should try to solve all of your problems
with single-hidden-layer networks. 

### 2 Activation Functions
Activation functions decide whether a neuron should be activated or not by calculating the weighted sum and further adding bias with it. They are differentiable operators to transform input signals to outputs

### 3 Implementation of Multilayer Perceptron 
So we can think of this as a classification dataset with 784 input features and 10 classes.
"""


class Neuron:
    """
    The hidden layers and the output layer Neurons are where the weights and biases come into play.
    Each neuron in these layers has associated weights that determine the strength and influence
    of its connections with the neurons in the previous layer.
    The weights are used to scale the input values and adjust the importance of each neuron's contribution to the final
    output.
    """

    def __init__(self, weights: list = None, feature_x_neuron_output: float = None, bias: float = 0.1):

        self.weights = weights
        self.bias = bias
        self.feature_x_neuron_output = feature_x_neuron_output
        self.delta = None

    def update_weights(self, weights: []):
        self.weights = weights

    def update_feature_x_neuron_output(self, feature_x_neuron_output: float):
        self.feature_x_neuron_output = feature_x_neuron_output


class NeuralNetwork:
    # initialise the neural network
    def __init__(self):
        self.matrix_n00b = MatrixNoobNp()
        self.layers = []

    def set_input_layer(self, features: List):
        """
        _summary_ 
        input layer means list of features for the neural network
        Pay attention current input layer neurons does not have any parameters, only features.
        Consider input layer as a VECTOR features 
        """
        input_layer_neurons = [Neuron(feature_x_neuron_output=f) for f in features]
        if len(self.layers) == 0:
            self.layers.append(input_layer_neurons)
        else:
            self.layers[0] = input_layer_neurons

    def add_layer(self, neurons_in_layer: int):
        # check if we have input layer
        if self.layers.__len__() > 0:
            hidden_layers_neurons = []
            for _ in range(neurons_in_layer):
                # initialize weights with gaussian distribution
                left_hand_side_layer_amount_features_x = len(self.layers[-1])
                weights_gauss = self.weight_gaussian_distribution(left_hand_side_layer_amount_features_x)
                hidden_layers_neurons.append(Neuron(weights=weights_gauss))

            self.layers.append(hidden_layers_neurons)

    @staticmethod
    def activation_relu(x: float):
        return max(0, x)

    @staticmethod
    def activation_sigmoid(x: float):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def calculate_loss(output, expected_output):
        """_summary_
        Mean Squared Error (MSE) loss. 
        It measures the difference between the predicted output (output) of the neural network 
        and the expected output (expected_output) for a given set of inputs.
        """
        return 0.5 * np.sum((expected_output - output) ** 2)

    @staticmethod
    def weight_gaussian_distribution(left_hand_side_layer_amount_features_x: int) -> list:
        """_summary_
                They sample the weights from a normal probability distribution centered around zero
        and with a standard deviation that is related to the number of incoming links into a node
        1/√(number of incoming links).
        In other words weights 'w' will be initialized with 67% probability in normal distribution window

        See https://www.techtarget.com/whatis/definition/normal-distribution

        Args:
            hidden_nodes (int): _description_
            input_nodes (int): _description_

        Returns:
            _type_: _description_
        """
        mean = 0
        stddev = 0.1
        return np.random.normal(mean, stddev, (left_hand_side_layer_amount_features_x)).tolist()

    def forward_propagation(self):
        input_data = self.layers[0]
        layers = self.layers[1:]

        input_features = [[n.feature_x_neuron_output for n in input_data]]

        # Calculate forvard signal based on matrices multiplication algorithm
        layers_nodes_output = []
        for layer in layers:
            weights_nodes_in_layer = []
            for node in layer:
                weights_nodes_in_layer.append(node.weights)

            res = self.matrix_n00b.matrices_multiplication(input_features,
                                                           self.matrix_n00b.transpose(weights_nodes_in_layer))
            layers_nodes_output.append(res[0])
            input_features = res

        # Use activation function for calulation outputs for the each node
        for i, layer in enumerate(layers_nodes_output):
            for j, output in enumerate(layer):
                layers[i][j].feature_x_neuron_output = self.activation_sigmoid(
                    output)

    def backward_propagation(self, expected_output_target: list, learning_rate: float):
        """
        backward propagation (more commonly called backpropagation)
        Error is calculated between the expected outputs and the outputs forward propagated from the network. 

        Args:
            target_outputs: (list) actual values from dataset
        """
        # learning_rate = 0.1
        # input_data = self.layers[0]
        # output_layer = self.layers[-1]

        """
        Backward propagation algorithm for output layer
        https://www.youtube.com/watch?v=Ilg3gGewQ5U&t=3s&ab_channel=3Blue1Brown
        """
        # final_outputs = [n.feature_x_neuron_output for n in output_layer]
        # hidden_outputs = [n.feature_x_neuron_output for n in hidden_layers[-1]]

        # Calculate delta errors for the hidden layers

        # delta_err_output = []
        delta_error_weights_matrix = None
        delta_errors_by_layers = []
        for i in range(len(self.layers)-1, 0, -1):
            right_hand_side_layer = self.layers[i]
            left_hand_side_layer = self.layers[i-1]

            right_hand_side_layer_outputs = [
                n.feature_x_neuron_output for n in right_hand_side_layer]
            left_hand_side_layer_outputs = [
                n.feature_x_neuron_output for n in left_hand_side_layer]

            # convert left_hand_side_layer_outputs to the matrix
            left_hand_side_layer_outputs_matrix = self.matrix_n00b.list_to_matrix(
                left_hand_side_layer_outputs, len(left_hand_side_layer_outputs))
            right_hand_side_activations_matrix = self.matrix_n00b.list_to_matrix(
                right_hand_side_layer_outputs, len(right_hand_side_layer_outputs))

            # if output layer
            if i == len(self.layers[:-1]):
                # left hand side weights -> right hand side output error
                # hidden_errors = numpy.dot(self.who.T, output_errors)
                delta_err_output = self.calculate_output_delta(output_actual=expected_output_target,
                                                               output_predicted=right_hand_side_layer_outputs)
                delta_err_output = self.matrix_n00b.list_to_matrix(
                    delta_err_output, len(delta_err_output))

            # if hidden layer
            elif i > 1:
                # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
                # delta_error_weights_matrix, left_hand_side_activations, right_hand_side_weights):

                right_hand_side_weights = [
                    n.weights for n in right_hand_side_layer]
                delta_err_output = self.calculate_hidden_delta(delta_error_weights_matrix,
                                                               left_hand_side_layer_outputs,
                                                               right_hand_side_weights)
            # if input layer
            else:
                # self.wih += self.lr * numpy.dot(
                #    (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                #    numpy.transpose(inputs))
                weights_of_first_hidden_layer = [
                    n.weights for n in right_hand_side_layer]
                delta_err_output = self.calculate_input_delta(delta_error_weights_matrix=delta_error_weights_matrix,
                                                              input_features=left_hand_side_layer_outputs,
                                                              weights_of_first_hidden_layer=weights_of_first_hidden_layer)

            # Calculate errors for the right_hand_side_layer_outputs
            delta_error_matrix_T = self.matrix_n00b.transpose(delta_err_output)

            # calculate gradient for the weights in the last layer
            delta_error_weights_matrix = self.matrix_n00b.matrices_multiplication(delta_error_matrix_T,
                                                                                  left_hand_side_layer_outputs_matrix)
            delta_errors_by_layers.append(delta_error_weights_matrix)

        """
        STEP 
        Update weights for the node
        """
        layers_skeep_input_layer = self.layers[1:]
        layers_skeep_input_layer.reverse()
        for i, layer in enumerate(layers_skeep_input_layer):
            for j, node in enumerate(layer):
                res = learning_rate * np.array(delta_errors_by_layers[i][j])
                node.weights = (np.array(node.weights) +
                                np.array(res)).tolist()

        """
        STEP 5 
        Calculate error and update weights for hidden layers
        """

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def calculate_output_delta(self, output_actual, output_predicted):
        diff = (np.array(output_actual) - np.array(output_predicted))
        delta_output = diff * \
            self.sigmoid_derivative(np.array(output_predicted))
        return delta_output.tolist()

    def calculate_hidden_delta(self, delta_error_weights_matrix, left_hand_side_activations, right_hand_side_weights):

        #        delta_output_matrix = self.matrix_n00b.list_to_matrix(delta_output, len(delta_output))
        #        right_hand_side_weights_T = self.matrix_n00b.transpose(right_hand_side_weights)

        delta_output_res = self.matrix_n00b.matrices_multiplication(
            delta_error_weights_matrix, right_hand_side_weights)

        delta_hidden = np.array(
            delta_output_res) * self.sigmoid_derivative(np.array(left_hand_side_activations))
        return self.matrix_n00b.list_to_matrix(delta_hidden.tolist(), len(delta_hidden.tolist()))

    def calculate_input_delta(self, delta_error_weights_matrix, input_features, weights_of_first_hidden_layer):
        """
        Calculate the delta for the input layer based on the delta from the output layer and weights of the first hidden layer.

        :param delta_output: List of delta values from the output layer.
        :param input_features: List of input features to the network.
        :param weights_of_first_hidden_layer: Matrix representing the weights connecting the input layer to the first hidden layer.
        :return: List representing the delta values for the input layer.
        """

        # Multiply the delta_output_matrix with the weights_of_first_hidden_layer to get the resulting matrix
        # representing the effect of the output delta values on the input layer before considering activation function derivative.

        delta_output_res = self.matrix_n00b.matrices_multiplication(
            delta_error_weights_matrix, weights_of_first_hidden_layer)

        # Element-wise multiplication of the first row of the resulting matrix (delta_output_res[0])
        # with the derivative of the sigmoid function of the first input feature.
        # This step incorporates the effect of the activation function on the input layer's delta values.
        input_to_hiden_delta = np.array(
            delta_output_res[0]) * self.sigmoid_derivative(np.array(input_features[0]))

        # Convert the NumPy array back to a list and return.
        return self.matrix_n00b.list_to_matrix(input_to_hiden_delta.tolist(), len(input_to_hiden_delta.tolist()))

    def train(self, inputs_list, targets_list, learning_rate, epoch):

        for e in range(epoch):
            print('@@@@@@@ epoch -> ', e)
            for i in range(len(inputs_list)):
                print(' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> data samle -> ', i)
                self.set_input_layer(inputs_list[i])
                self.forward_propagation()
                self.backward_propagation(
                    expected_output_target=targets_list[i], learning_rate=learning_rate)

    def pred(self, input):
        self.set_input_layer(input)
        self.forward_propagation()
        output_lauer_outputs = [
            n.feature_x_neuron_output for n in self.layers[-1]]
        return output_lauer_outputs

    def calculate_accuracy(self, predicted, target):
        if len(predicted) != len(target):
            raise ValueError(
                "The predicted and target lists must have the same length.")

        correct_predictions = 0
        for i in range(len(predicted)):
            if predicted[i].index(max(predicted[i])) == target[i].index(max(target[i])):
                correct_predictions += 1

        total_predictions = len(predicted)
        accuracy = correct_predictions / total_predictions
        return accuracy

    def print_nn(self):
        print("print your nn:")

        if len(self.layers) == 0:
            print("Network is empty")
            return

        input_layer = self.layers[0]
        hidden_layers = self.layers[1:]

        print("@@@ input layer:")
        for index, node in enumerate(input_layer):
            print(
                f"index -> {index}, neuron -> {node}, feature -> {node.feature_x_neuron_output}")

        for index, layer in enumerate(hidden_layers):
            print(f"@@@ layer № {index+1} @@@")
            for index, node in enumerate(layer):
                print(
                    f"index -> {index}, neuron -> {node}, weights -> {len(node.weights)}, weight data -> {node.weights}")
                # print(f"Neuron output x -> {node.feature_x_neuron_output}")

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def load_model(self, filename):
        with open(filename, 'rb') as file:
            loaded_nn = pickle.load(file)
            return loaded_nn


# TESTS
class NeuralNetworkBasicTest(unittest.TestCase):

    # Helper methods
    def read_mnist_data(self, output_nodes_amount: int, csv_path: str):
        with open(csv_path, "r") as training_data_file:
            training_data_list = training_data_file.readlines()
        # TRAIN -> go through all records in the training data set
        inputs_list = []
        targets_list = []
        for record in training_data_list[1:]:
            # split the record by the ',' commas
            all_values = record.split(",")
            # scale and shift the inputs
            input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            target = (np.zeros(output_nodes_amount) + 0.01)
            # all_values[0] is the target label for this record
            target[int(all_values[0])] = 0.99

            inputs_list.append(input.tolist())
            targets_list.append(target.tolist())

        return inputs_list[1:50], targets_list[1:50]

    def read_mnist_train_data(self):
        return self.read_mnist_data(output_nodes_amount=10, csv_path="./data/mnist_test_10k.csv")

    def read_mnist_test_data(self):
        return self.read_mnist_data(output_nodes_amount=10, csv_path="./data/mnist_test_10k.csv")

    def setUp(self):
        self.nn = NeuralNetwork()

    def test_set_input_layer(self):
        self.nn.set_input_layer(features=[1, 2])
        self.nn.print_nn()
        self.assertEqual(len(self.nn.layers[0]), 2)

    def test_add_one_layer(self):
        self.nn.set_input_layer(features=[1, 2])
        self.nn.add_layer(neurons_in_layer=3)
        self.nn.print_nn()
        self.assertEqual(len(self.nn.layers[1]), 3)

    def test_add_two_layers(self):
        self.nn.set_input_layer(features=[1, 2])
        self.nn.add_layer(neurons_in_layer=3)
        self.nn.add_layer(neurons_in_layer=2)
        self.nn.print_nn()
        self.assertEqual(len(self.nn.layers[2]), 2)

    def test_forward_propagation(self):
        self.nn.set_input_layer(features=[1, 2])
        self.nn.add_layer(neurons_in_layer=3)
        self.nn.add_layer(neurons_in_layer=4)
        self.nn.add_layer(neurons_in_layer=1)
        self.nn.forward_propagation()
        self.assertIsNotNone(self.nn.layers[2][1].feature_x_neuron_output)

    def test_back_propagation(self):
        expected_data_10 = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        expected_data_2 = [1.1, 2.2]
        self.nn.set_input_layer(features=expected_data_2)
        self.nn.add_layer(neurons_in_layer=4)
        # TODO uncomment me and fix architecture more than 2 layers
        # self.nn.add_layer(neurons_in_layer=4)
        self.nn.add_layer(neurons_in_layer=2)
        self.nn.forward_propagation()
        self.nn.backward_propagation(
            expected_output_target=expected_data_2, learning_rate=0.01)

    def test_train_handwritten_digit_recognition(self):
        inputs_list, targets_list = self.read_mnist_train_data()
        self.nn.set_input_layer(features=inputs_list[0])
        self.nn.add_layer(neurons_in_layer=100)
        self.nn.add_layer(neurons_in_layer=10)

        self.nn.train(inputs_list, targets_list, learning_rate=0.01, epoch=1)
        self.nn.save_model("./trained_nn.pkl")

    def test_pred_handwritten_digit_recognition(self):
        # 10k trainig dataset:
        # predicted vector   ->  [0.13, 0.86, 0.55, 0.45, 0.25, 0.35, 0.57, 0.24, 0.39, 0.43]
        # real target vector ->  [0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

        inputs_list, targets_list = self.read_mnist_test_data()
        loaded_nn = self.nn.load_model("./trained_nn.pkl")

        predicted_outputs = []
        for i in range(len(inputs_list)):
            predicted_vector = loaded_nn.pred(inputs_list[i])
            formatted_predicted_vector = [
                float(f'{num:.2f}') for num in predicted_vector]
            print("predicted vector   -> ", formatted_predicted_vector)
            print("real target vector -> ", targets_list[i])

            predicted_outputs.append(predicted_vector)

        res = loaded_nn.calculate_accuracy(
            predicted_outputs[:10], targets_list[:10])
        # 42k damples = Accuracy =  0.9
        print("Accuracy = ", res)


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()
    # suite.addTest(NeuralNetworkBasicTest('test_set_input_layer'))
    # suite.addTest(NeuralNetworkBasicTest('test_add_one_layer'))
    # suite.addTest(NeuralNetworkBasicTest('test_add_two_layers'))
    # suite.addTest(NeuralNetworkBasicTest('test_forward_propagation'))
    # suite.addTest(NeuralNetworkBasicTest('test_back_propagation'))

    suite.addTest(NeuralNetworkBasicTest('test_train_handwritten_digit_recognition'))
    suite.addTest(NeuralNetworkBasicTest('test_pred_handwritten_digit_recognition'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # TODO questions 'check notations and formulas' -> https://d2l.ai/chapter_multilayer-perceptrons/backprop.html