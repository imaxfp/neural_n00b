import unittest
from typing import List
from collections import deque
import numpy as np
import math
import pickle
from src.matrix.matrix_n00b_np import MatrixNoobNp


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
        #output of the neuron, can be a featore for the input to the lext layer 
        self.feature_x_neuron_output = feature_x_neuron_output
        #delta error 
        self.delta = None

    def update_weights(self, weights: []):
        self.weights = weights

    def update_feature_x_neuron_output(self, feature_x_neuron_output: float):
        self.feature_x_neuron_output = feature_x_neuron_output


class NeuralNetwork:
    '''
    Neural net custom implementation
    '''
    def __init__(self):
        self.matrix_n00b = MatrixNoobNp()
        self.layers = []

    def get_layers(self):
        return self.layers    

    def add_layer(self, neurons_in_layer: int):
            #add initial layer. FEATURES layer
            if len(self.layers) == 0:
                input_features_layer=[]
                for i in range(0, neurons_in_layer):
                    input_features_layer.append(0.1)
                self.layers.append(input_features_layer)    
            
            #add hidden layer
            else:
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
        # prevent the overflow error, you can modify the function to handle extreme values of x:
        if x > 0:
            return 1 / (1 + math.exp(-x))
        else:
            z = math.exp(x)
            return z / (1 + z)

    @staticmethod
    def calculate_loss(output, expected_output):
        """
        Mean Squared Error (MSE) loss. 
        It measures the difference between the predicted output (output) of the neural network 
        and the expected output (expected_output) for a given set of inputs.
        """
        return 0.5 * np.sum((expected_output - output) ** 2)

    @staticmethod
    def weight_gaussian_distribution(left_hand_side_layer_amount_features_x: int) -> list:
        """
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

    def forward_propagation(self, input_data):
        """
        Forward Propagation in Neural Networks:
        The main idea behind forward propagation is to pass the input data through each layer of the neural network (from left to rigth by layers) 
        successively transforming it using WEIGHTS and ACTIVATION functions. The final output is a prediction based on the
        transformed input. This process involves:
        1. Multiplying the input by the layer's weights (matrix multiplication).
        2. Adding biases to the result. (will be added)
        3. Applying an activation function to introduce non-linearity.
        This is done for each layer until the final layer produces the network's output.
        """
        print("@@@ - forward propagation:")

        layers = self.layers[1:]

        # Calculate forvard signal based on matrices multiplication algorithm
        layers_nodes_output = []
        for layer in layers:
            weights_nodes_in_layer = []
            for node in layer:
                weights_nodes_in_layer.append(node.weights)

            input_data_matrix = self.matrix_n00b.list_to_matrix(input_data, len(input_data))    

            res = self.matrix_n00b.matrices_multiplication(input_data_matrix,
                                                           self.matrix_n00b.transpose(weights_nodes_in_layer))
            layers_nodes_output.append(res[0])
            input_data = res[0]

        # Use activation function for calulation outputs for the each node
        for i, layer in enumerate(layers_nodes_output):
            for j, output in enumerate(layer):
                layers[i][j].feature_x_neuron_output = self.activation_sigmoid(output)

    def backward_propagation(self, expected_output_target: list, learning_rate: float):
        """
        Backward Propagation in Neural Networks:
        The primary purpose of backward propagation (often termed 'backprop') is to adjust the weights and biases of the neural network 
        in response to the error in the network's predictions. It involves:
        1. Computing the gradient of the loss function with respect to each weight by applying the chain rule, which measures how much 
        each weight contributed to the error.
        2. Adjusting the weights and biases in the opposite direction of the gradient to minimize the error.
        3. Propagating this error backwards through the network, layer by layer, from the last layer to the first (right to left approach).
        By repeatedly applying forward and backward propagation in training, the neural network learns the optimal weights and biases 
        that best map inputs to desired outputs.

        https://www.youtube.com/watch?v=Ilg3gGewQ5U&t=3s&ab_channel=3Blue1Brown
        """
        # Calculate backward propagation error deltas for the ouput layer
        delta_errors_by_layers_lifo = self.back_propagation_errors_calculation(expected_output_target)
        self.back_propagation_update_weights(delta_errors_by_layers_lifo, learning_rate)



    def back_propagation_errors_calculation(self, expected_output_target):        
        print("@@@ -> back_propagation_errors_calculation")
        
        right_hand_side_layer_outputs = [n.feature_x_neuron_output for n in self.layers[-1]]
            
        #Calculate errors for the output layer
        errors_all_layers_lifo = []
        errors_curent = []
        print("Current layer is output layer ", len(self.layers))
        for i in range(0, len(expected_output_target)):
            diff = expected_output_target[i] - right_hand_side_layer_outputs[i]
            errors_curent.append(np.array(diff))
        errors_all_layers_lifo.append(errors_curent)    
        
        #Calculate errors for the hidden layers
        for i in range(1, len(self.layers)):
            #Pay attention 'self.layers[-i]' means steps from right to left by layers (left_hand_side <- right_hand_side)
            weights_output_layer = [n.weights for n in self.layers[-i]]
            
            weigths_matrix = np.array(weights_output_layer).T
            error_matrix = np.array(errors_all_layers_lifo[-1])
            print("Current layer is hidden layer ",len(self.layers) - i)
            print("weigths_matrix {w_m} * error_matrix {e_m}".format(w_m=weigths_matrix.shape, e_m=error_matrix.shape) )

            hidden_errors = np.dot(weigths_matrix, error_matrix)    
            hidden_errors = [np.array(x) for x in hidden_errors]
            errors_all_layers_lifo.append(hidden_errors)

        errors_all_layers_lifo.reverse()
        return errors_all_layers_lifo
    
    def back_propagation_update_weights(self, delta_errors_by_layers_lifo, learning_rate):
        print("@@@ -> back_propagation_update_weights")
        for i in range(1, len(self.layers)):            
            
            right_hand_side_outputs = np.array([n.feature_x_neuron_output for n in self.layers[-i]])

            #If hidden layer (Neurons)                    
            if type(self.layers[-(i+1)][0]) is Neuron:                
                left_hand_side_outputs = np.array([n.feature_x_neuron_output for n in self.layers[-(i+1)]], ndmin=2)            
            #if inptup layer, in other words input features                 
            else:
                left_hand_side_outputs = np.array([n for n in self.layers[-(i+1)]], ndmin=2)
                                
            output_errors = delta_errors_by_layers_lifo[-i]

            #TODO separate functions implementation for different derivatives. 
            # Calc error for the each node.
            derivative_err = (output_errors * right_hand_side_outputs * (1.0 - right_hand_side_outputs))
            # Derivative_err is array of np.array for the matrix multiplication algorithm. 
            derivative_err = np.array([np.array([e]) for e in output_errors])
                              

            print("Current layer is hidden layer ",len(self.layers) - i)
            print("derivative_err {d_err} * error_matrix {l_h_out}".format(d_err=derivative_err.shape, l_h_out=left_hand_side_outputs.shape) )

            err_weights_matrix = learning_rate * np.dot(derivative_err, left_hand_side_outputs)

            #Update neuron weigths 
            for j in range(0, len(err_weights_matrix)):
                neuron = self.layers[-i][j]
                old_weights = np.array(neuron.weights)
                updated_weights = old_weights + err_weights_matrix[j]
                neuron.weights = updated_weights.tolist() 
                    

# TODO implemet different derivatives 'activation function derivative during back prop'
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    

    def check_input_features(self, inputs_list):
        # Check the length of x_features against the length of self.layers[0]
        if len(inputs_list) != len(self.layers[0]):
            raise ValueError("Length mismatch between x_features and the first layer.")

        # Check types of values in x_features
        for feature in inputs_list:
            if not isinstance(feature, (int, float)):
                raise TypeError(f"Feature {feature} is not of type int or float.")


    def train(self, inputs_list, targets_list, learning_rate, epoch):
        for e in range(epoch):
            print('@@@@@@@@@@@@@@@@@@@@@  epoch -> ', e)
            for i in range(len(inputs_list)):
                self.check_input_features(inputs_list[i])
                print(' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> data samle -> ', i)
                self.forward_propagation(input_data=inputs_list[i])
                self.backward_propagation(expected_output_target=targets_list[i], learning_rate=learning_rate)

    def pred(self, input):
        self.check_input_features(input)
        self.forward_propagation(input_data=input)
        output_lauer_outputs = [n.feature_x_neuron_output for n in self.layers[-1]]
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
    
    #TODO build confusion matrix 

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
                f"index -> {index}, feature -> {node}")

        for index, layer in enumerate(hidden_layers):
            print(f"@@@ layer № {index+1} @@@")
            for index, node in enumerate(layer):
                formatted_weights = ", ".join([f"{w:.3f}" for w in node.weights])
                print(
                    f"index -> {index}, neuron -> {node}, weights -> {len(node.weights)}, weight data -> {formatted_weights}")

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def load_model(self, filename):
        with open(filename, 'rb') as file:
            loaded_nn = pickle.load(file)
            return loaded_nn


# TESTS
class NeuralNetworkBasicTest(unittest.TestCase):

    def setUp(self):
        current_test = self._testMethodName
        print(f" ===============> Running test: {current_test} <===============") 
        self.nn = NeuralNetwork()

    def startTest(self, test):
        super().startTest(test)
        print(f"\n######## Running test: ######## {test._testMethodName}\n{'='*40}")
        
    def test_set_input_layer(self):
        self.nn.add_layer(neurons_in_layer=4)
        self.nn.print_nn()
        self.assertEqual(len(self.nn.layers[0]), 4)

    def test_add_two_layers(self):
        self.nn.add_layer(neurons_in_layer=2)
        self.nn.add_layer(neurons_in_layer=3)
        self.nn.print_nn()
        self.assertEqual(len(self.nn.layers[1]), 3)

    def test_forward_propagation(self):
        self.nn.add_layer(neurons_in_layer=2)
        self.nn.add_layer(neurons_in_layer=3)
        self.nn.add_layer(neurons_in_layer=4)
        self.nn.add_layer(neurons_in_layer=1)
        self.nn.forward_propagation(input_data=[1,2])
        self.assertTrue(type(self.nn.layers[2][1].feature_x_neuron_output) is float)
        self.assertEqual(len(self.nn.layers[3]), 1)

    def test_back_propagation(self):
        self.nn.add_layer(neurons_in_layer=9)
        self.nn.add_layer(neurons_in_layer=4)
        self.nn.add_layer(neurons_in_layer=2)
        self.nn.forward_propagation(input_data=[1,2,3,4,5,6,7,8,9])
        self.nn.backward_propagation(expected_output_target=[1.1, 2.2], learning_rate=0.01)

    def test_back_propagation_3_layers(self):
        self.nn.add_layer(neurons_in_layer=9)
        self.nn.add_layer(neurons_in_layer=100)
        self.nn.add_layer(neurons_in_layer=50)
        self.nn.add_layer(neurons_in_layer=27)
        self.nn.add_layer(neurons_in_layer=2)
        self.nn.forward_propagation(input_data=[1,2,3,4,5,6,7,8,9])
        self.nn.backward_propagation(expected_output_target=[1.1, 2.2], learning_rate=0.01)
        
    def test_train(self):
        self.nn.add_layer(neurons_in_layer=10)
        self.nn.add_layer(neurons_in_layer=100)
        self.nn.add_layer(neurons_in_layer=50)
        self.nn.add_layer(neurons_in_layer=25)
        self.nn.add_layer(neurons_in_layer=2)

        input=[0,1,2,3,4,5,6,7,8,9]
        input_data_set=[input, input, input]

        target=[0.1, 0.99]
        target_data_set=[target, target, target]

        self.nn.train(inputs_list=input_data_set, targets_list=target_data_set, learning_rate=0.01, epoch=10)

    def test_train_pred(self):
        self.nn.add_layer(neurons_in_layer=10)
        self.nn.add_layer(neurons_in_layer=50)
        self.nn.add_layer(neurons_in_layer=2)

        input=[0,1,2,3,4,5,6,7,8,9]
        input_data_set=[input, input, input]

        target=[0.1, 0.99]
        target_data_set=[target, target, target]

        self.nn.train(inputs_list=input_data_set, targets_list=target_data_set, learning_rate=0.01, epoch=10)

        res = self.nn.pred(input=input)   
        print("Prediction result => ", res)
        self.assertTrue(res[0]<res[1])     
        



if __name__ == '__main__':
    suite = unittest.TestSuite()
    #suite.addTest(NeuralNetworkBasicTest('test_set_input_layer'))    
    #suite.addTest(NeuralNetworkBasicTest('test_add_two_layers'))
    #suite.addTest(NeuralNetworkBasicTest('test_forward_propagation'))
    #suite.addTest(NeuralNetworkBasicTest('test_back_propagation'))
    #suite.addTest(NeuralNetworkBasicTest('test_back_propagation_3_layers'))

    #suite.addTest(NeuralNetworkBasicTest('test_train'))
    suite.addTest(NeuralNetworkBasicTest('test_train_pred'))


    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)