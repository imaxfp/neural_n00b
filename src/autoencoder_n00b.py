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
        for idx in range(len(layers)-1, -1, -1):
            super().add_layer(neurons_in_layer=layers[idx])

    def encode(slef):
        """
        The encoder's job is to transform the input data (often high-dimensional) into a lower-dimensional representation, 
        commonly referred to as the "bottleneck" or "latent space". This compressed representation captures the most salient features of the input data.

        The main idea behind embeddings is to capture semantic or relational meaning in a dense, lower-dimensional space.

        1. Dimensionality Reduction:
        Many discrete objects, especially words, can be associated with a high-dimensional space. 
        For example, a one-hot encoded representation of a vocabulary of 10,000 words requires 10,000 dimensions. 
        Embeddings map these objects to a much lower-dimensional space (e.g., 50, 100, or 300 dimensions)
        making the representations more manageable and computationally efficient.

        2. Semantic Relationships: 
        The geometry of the embedding space is designed such that semantically or functionally similar items are closer together. 
        For instance, in a word embedding space, words like "king" and "queen" might be closer together,
        whereas words like "king" and "apple" would be further apart.

        3. Transfer Learning and Pre-trained Embeddings: 
        Embeddings, especially word embeddings like Word2Vec or GloVe, can be pre-trained on large corpora and then used or fine-tuned on specific tasks.
        This allows for the transfer of knowledge from one task (like language modeling) to another (like sentiment analysis).

        4. Generalization:
        Since embeddings capture semantic meanings, they can help models generalize better. 
        If a model learns that "cat" and "kitty" are similar from their embeddings, 
        it can infer that information or sentiments about "cat" might apply to "kitten" as well.

        5. Embeddings Beyond Words: 
        While word embeddings are the most widely known, embeddings can be used for various types of data. For example:
            - User and item embeddings for recommendation systems.
            - Graph embeddings to capture the structure of graphs.
            - Entity embeddings for categorical variables in tabular data.
        6. Training: 
        Embeddings can be trained in several ways:
            - Supervised: As part of a bigger model for a specific task (e.g., sentiment analysis).
            - Unsupervised: Using algorithms like Word2Vec, GloVe, or FastText which leverage large corpora to learn word relationships based on context.
            - Semi-supervised or Transfer Learning: Starting with pre-trained embeddings and fine-tuning them on a specific task.    
        """

        return None
    

    def decode(self, embedding):

        return None



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

        #Check if oupot layer has the same length as imput layer
        self.assertEqual(len(self.nn_autoencoder.get_layers()[-1]), layers[0])
        self.assertEqual(len(self.nn_autoencoder.get_layers()[-2]), layers[1])

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(NeuralNetworkBasicTest('test_create_autoencoder'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)