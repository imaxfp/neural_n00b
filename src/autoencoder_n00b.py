import unittest
from typing import List
from collections import deque
import numpy as np
import math
from perceptron_n00b import NeuralNetwork


class Autoencoder(NeuralNetwork):
    """
    """

    def __init__():
        super().__init__() 

    def create_autoencoder(layers: List[int]):
        """
        list: where size list is amount of layers for encoder. Values of a list is number of the neurons in the each layer 
        """

        #Create ENCODER
        for e in layers:
                 

        #Create DECODER


    @staticmethod
    def calculate_loss(output, expected_output):        
        """
        Mean Squared Error (MSE) loss. 
        It measures the difference between the predicted output (output) of the neural network 
        and the expected output (expected_output) for a given set of inputs.
        """
        #TODO
        return 0.5 * np.sum((input - output) ** 2)        


# TESTS
class NeuralNetworkBasicTest(unittest.TestCase):

    def setUp(self):
        self.nn_autoencoder = Autoencoder()

    def startTest(self, test):
        super().startTest(test)
        print(f"\n######## Running test: ######## {test._testMethodName}\n{'='*40}")
        
    def test_create_autoencoder(self):
        self.nn_autoencoder.set_input_layer(features=[1, 2, 4, 5, 6, 7])
        self.nn_autoencoder.print_nn()
        self.assertEqual(len(self.nn.layers[0]), 2)

  


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(NeuralNetworkBasicTest('test_create_autoencoder'))
    suite.addTest(NeuralNetworkBasicTest('test_create_autoencoder'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)