import unittest
from dataset_service import read_mnist_data, read_mnist_test_data, read_mnist_train_data
from perceptron_n00b import NeuralNetwork

class NeuralNetworkBasicTest(unittest.TestCase):    
    def setUp(self):
        self.nn = NeuralNetwork()

    def test_train_handwritten_digit_recognition(self):
        # Read train and test data 
        train_path = "./data/mnist_train_60k.csv.zip"
        train_inputs, train_targets = read_mnist_data(output_nodes_amount=10,
                                                        samples=150,
                                                        csv_path=train_path)


        #create perceptron NN
        self.nn.set_input_layer(features=train_inputs[0])
        self.nn.add_layer(neurons_in_layer=100)
        self.nn.add_layer(neurons_in_layer=10)

        #train the model
        self.nn.train(train_inputs, train_targets, learning_rate=0.1, epoch=5)
        self.nn.save_model("./models/perceptron_n00b_10_epoch_150_samples.pkl")


if __name__ == '__main__':    
    suite = unittest.TestSuite()
    suite.addTest(NeuralNetworkBasicTest('test_train_handwritten_digit_recognition'))

    runner = unittest.TextTestRunner()
    runner.run(suite)

    # TODO questions 'check notations and formulas' -> https://d2l.ai/chapter_multilayer-perceptrons/backprop.html