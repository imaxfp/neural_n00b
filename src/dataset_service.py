import numpy as np

def read_mnist_data(output_nodes_amount: int, csv_path: str):
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

    return inputs_list, targets_list
        
def read_mnist_train_data(samples=200):
    inputs_list, targets_list = read_mnist_data(output_nodes_amount=10, csv_path="./data/mnist_test_10k.csv")
    return inputs_list[1:samples], targets_list[1:samples]

def read_mnist_test_data(samples=20):
    inputs_list, targets_list = read_mnist_data(output_nodes_amount=10, csv_path="./data/mnist_test_10k.csv")
    return inputs_list[1:samples], targets_list[1:samples]
