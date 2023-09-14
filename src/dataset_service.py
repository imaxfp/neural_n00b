import numpy as np
import pandas as pd
import zipfile
import zipfile
import csv

def read_mnist_data(output_nodes_amount: int, samples: int, csv_path: str):

    inputs_list = []
    targets_list = []

    # Open the ZIP file
    with zipfile.ZipFile(csv_path, 'r') as z:
        # Extract the name of the first file inside the ZIP
        csv_name = z.namelist()[0]
        
        with z.open(csv_name) as f:
            reader = csv.reader(line.decode('utf-8') for line in f)

            # Skip the first line
            print("The header of the CSV -> ", next(reader, None))
            for i, row in enumerate(reader):
                if i >= samples:
                    break
                input = (np.asfarray(row[1:]) / 255.0 * 0.99) + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                target = (np.zeros(output_nodes_amount) + 0.01)
                # all_values[0] is the target label for this record
                target[int(row[0])] = 0.99

                inputs_list.append(input.tolist())
                targets_list.append(target.tolist())

    return inputs_list, targets_list    