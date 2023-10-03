import unittest
from dataset_service import read_mnist_data
from perceptron_n00b import NeuralNetwork

# Train perceprtron model for the hand writen digits recognition. 

class NeuralNetworkBasicTest(unittest.TestCase):    
    def setUp(self):
        self.nn = NeuralNetwork()

    def test_pred_handwritten_digit_recognition(self):

        # predicted vector   ->  [0.13, 0.86, 0.55, 0.45, 0.25, 0.35, 0.57, 0.24, 0.39, 0.43]
        # real target vector ->  [0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

        train_path = "./data/mnist_test_10k.csv.zip"
        test_inputs, test_targets = read_mnist_data(output_nodes_amount=10,
                                                        samples=150,
                                                        csv_path=train_path)

        loaded_nn = self.nn.load_model("./models/perceptron_n00b_5_epoch_15k_samples.pkl")

        predicted_outputs = []
        for i in range(len(test_inputs)):
            predicted_vector = loaded_nn.pred(test_inputs[i])
            formatted_predicted_vector = [
                float(f'{num:.2f}') for num in predicted_vector]
            print("predicted vector   -> ", formatted_predicted_vector)
            print("real target vector -> ", test_targets[i])

            predicted_outputs.append(predicted_vector)

        res = loaded_nn.calculate_accuracy(predicted_outputs, test_targets)
        print("Accuracy = ", res)
  

if __name__ == '__main__':    
    suite = unittest.TestSuite()

    # perceptron_n00b_20_epoch_100_samples.pkl
    # test = 1k samples  
    # Accuracy =  0.566

    # perceptron_n00b_10_epoch_1k_samples.pkl
    # test = 1k samples  
    # Accuracy =  0.692

    # perceptron_n00b_10_epoch_5k_samples.pkl
    # test = 1k samples  
    # Accuracy =  0.713

    # perceptron_n00b_20_epoch_5k_samples.pkl
    # test = 1k samples  
    # Accuracy =  0.768

    # Ran 1 test in 1196.205s
    # perceptron_n00b_5_epoch_15k_samples.pkl
    # test = 1k samples  
    # Accuracy =  0.688
    suite.addTest(NeuralNetworkBasicTest('test_pred_handwritten_digit_recognition'))

    runner = unittest.TextTestRunner()
    runner.run(suite)