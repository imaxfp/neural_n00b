import numpy
import numpy as np

# scipy.special for the sigmoid function expit()
import scipy.special

# library for plotting arrays
import matplotlib.pyplot


# neural network class definition
class neuralNetwork:
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)
        )
        self.who = numpy.random.normal(
            0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)
        )

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        # weight h o 'output' layer weights
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == "__main__":
    # number of input, hidden and output nodes
    # 28x28-pixel square, representing a handwritten digit (0 to 9).
    #input_nodes = 784
    input_nodes = 3
    # second layer 'hidden' layer
    #hidden_nodes = 100
    hidden_nodes = 4
    # prediction numbers from 0 to 9
    #output_nodes = 10
    output_nodes = 2

    # learning rate is 0.3
    # Learning rate is a critical hyperparameter in machine learning that determines the step size taken during parameter updates.

    # Low Learning Rate - The model makes small updates to the parameters at each iteration.
    # High Learning Rate - Large steps to update the parameters at each iteration.
    # Optimal Learning Rate - The model finds a balance, making updates that converge efficiently to the optimal solution.

    # (Adam, RMSprop) that automatically adjust the learning rate during training to strike a balance between convergence speed and stability.
    learning_rate = 0.3

    # Create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Prepare data for training process
    with open("./data/mnist_train_100.csv", "r") as training_data_file:
        training_data_list = training_data_file.readlines()
    # train the neural network

    # TRAIN -> go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(",")
        # scale and shift the inputs
        #inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        inputs = [1,2,3]
        # create the target output values (all 0.01, except the desired label which is 0.99)
        #targets = numpy.zeros(output_nodes) + 0.01
        targets = [0.1,0]
        # all_values[0] is the target label for this record
        #targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

    # TEST -> Verification (Test) check data outside as a customer
    with open("./data/mnist_test_10.csv", "r") as file_test_data:
        test_data_list = file_test_data.readlines()

    pred_list = []
    target_list = []
    for record in test_data_list:
        all_values = record.split(",")
        target_list.append(int(all_values[0]))
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        pred = n.query(inputs)
        # max value index presents predicted value
        max_value_index = np.argmax(pred)
        pred_list.append(max_value_index)

    from sklearn.metrics import accuracy_score

    # Calculate the accuracy metric
    # Tested with full dataset - Accuracy: 94.87%
    accuracy = accuracy_score(target_list, pred_list)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
