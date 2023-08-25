import unittest
from dataset_n00b import read_mnist_test_data, read_mnist_train_data
from nerual_n00b import NeuralNetwork

# TESTS
class NeuralNetworkBasicTest(unittest.TestCase):    
    def setUp(self):
        self.nn = NeuralNetwork()

    def test_train_handwritten_digit_recognition(self):
        inputs_list, targets_list = read_mnist_train_data()
        self.nn.set_input_layer(features=inputs_list[0])
        self.nn.add_layer(neurons_in_layer=100)
        self.nn.add_layer(neurons_in_layer=10)

        self.nn.train(inputs_list, targets_list, learning_rate=0.01, epoch=1)
        self.nn.save_model("./trained_nn.pkl")

    def test_pred_handwritten_digit_recognition(self):
        # 10k trainig dataset:
        # predicted vector   ->  [0.13, 0.86, 0.55, 0.45, 0.25, 0.35, 0.57, 0.24, 0.39, 0.43]
        # real target vector ->  [0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

        inputs_list, targets_list = read_mnist_test_data()
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
    suite.addTest(NeuralNetworkBasicTest('test_train_handwritten_digit_recognition'))
    #suite.addTest(NeuralNetworkBasicTest('test_pred_handwritten_digit_recognition'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # TODO questions 'check notations and formulas' -> https://d2l.ai/chapter_multilayer-perceptrons/backprop.html