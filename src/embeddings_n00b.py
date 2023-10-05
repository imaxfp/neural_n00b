import unittest
from typing import List
from collections import deque
import numpy as np
from perceptron_n00b import NeuralNetwork


class Autoencoder(NeuralNetwork):
    """
    """

    def __init__():
        super().__init__() 


@staticmethod
def calculate_loss(output, expected_output):
        """
        Mean Squared Error (MSE) loss. 
        It measures the difference between the predicted output (output) of the neural network 
        and the expected output (expected_output) for a given set of inputs.
        """
        #TODO
        return 0.5 * np.sum((input - output) ** 2)        

# TODO 
# Denoising 

            
# TESTS
class NeuralNetworkBasicTest(unittest.TestCase):

    def setUp(self):
        self.nn = EncoderDecoder()

    def startTest(self, test):
        super().startTest(test)
        print(f"\n######## Running test: ######## {test._testMethodName}\n{'='*40}")
        
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
        input_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        expected = [1.1, 2.2]
        self.nn.set_input_layer(features=input_data)
        self.nn.add_layer(neurons_in_layer=4)
        self.nn.add_layer(neurons_in_layer=2)
        self.nn.forward_propagation()
        self.nn.backward_propagation(expected_output_target=expected, learning_rate=0.01)

    def test_back_propagation_3_layers(self):
        input_data = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        expected = [1.1, 2.2]
        self.nn.set_input_layer(features=input_data)
        self.nn.add_layer(neurons_in_layer=100)
        self.nn.add_layer(neurons_in_layer=50)
        self.nn.add_layer(neurons_in_layer=27)
        self.nn.add_layer(neurons_in_layer=2)
        self.nn.forward_propagation()
        self.nn.backward_propagation(expected_output_target=expected, learning_rate=0.01)    

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(NeuralNetworkBasicTest('test_back_propagation'))
    suite.addTest(NeuralNetworkBasicTest('test_back_propagation_3_layers'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)