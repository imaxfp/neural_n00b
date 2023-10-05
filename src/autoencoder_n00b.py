import unittest
from typing import List
from collections import deque
import numpy as np
from src.perceptron_n00b import NeuralNetwork


class Autoencoder(NeuralNetwork):
    """
    """

    def __init__(self):
        super().__init__()

    def show_autoencoder_architecture(self):
        layers = super().get_layers()
        for id, layer in enumerate(layers):
            print(f"layer is {id}, amount of teurons is {len(layer)}")


    def create_autoencoder(self, layers: List[int]):
        """
        list: where size list is amount of layers for encoder. Values of a list is number of the neurons in the each layer 
        """
        #Create ENCODER
        for l in layers:
            super().add_layer(neurons_in_layer=l)
                 
        #Create DECODER
        layers.reverse() 
        for l in layers:
            super().add_layer(neurons_in_layer=l)

    def calculate_loss(self, output, expected_output):        
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
        layers = [784,200,100,50]
        self.nn_autoencoder.create_autoencoder(layers)
        self.nn_autoencoder.show_autoencoder_architecture()

        #self.assertEqual(len(self.nn.layers[0]), 2)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(NeuralNetworkBasicTest('test_create_autoencoder'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)