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

def read_zip_csv(csv_path, num_lines) -> pd:
    # Unzipping the file and reading the content
    with zipfile.ZipFile(csv_path, 'r') as z:
        # Assuming the first file in the zip is our target CSV
        with z.open(z.namelist()[0], 'r') as f:
            df = pd.read_csv(f, nrows=num_lines)
    return df

def read_and_clean_zip_csv(csv_path, num_lines) -> pd:
    # Unzipping the file and reading the content
    with zipfile.ZipFile(csv_path, 'r') as z:
        # Assuming the first file in the zip is our target CSV
        with z.open(z.namelist()[0], 'r') as f:
            df = pd.read_csv(f, nrows=num_lines)
            
    # Removing rows containing NaN values
    df_cleaned = df.dropna()

    # Storing the cleaned dataframe to a new zip file
    with zipfile.ZipFile(csv_path+"fixed_file.zip", 'w') as zipf:
        with zipf.open("cleaned_data.csv", 'w') as file:
            df_cleaned.to_csv(file, index=False)   

    return df_cleaned        


def mock_cat_dog_img_data(samples: int):
    # Mock data for CAT: more intensity in the center
    #Return mock images 2d matrices 4x4 for the 2 clases "cat", "dog"

    # Mock data for CAT: more intensity in the center
    X_cat_mock  =  [
        [[50, 50, 50, 50],
        [50, 200, 200, 50],
        [50, 200, 200, 50],
        [50, 50, 50, 50]],
    ]

    # Mock data for DOG: uniform intensity
    X_dog_mock = [
        [[150, 150, 150, 150],
        [150, 150, 150, 150],
        [150, 150, 150, 150],
        [150, 150, 150, 150]]
    ]

    # Introducing slight variations in the mock data for diversity
    for _ in range(samples-1):
        # Slight variation for CAT
        cat_variation = [
            [50 + np.random.randint(-10, 10) for _ in range(4)],
            [50 + np.random.randint(-10, 10)] + [200 + np.random.randint(-10, 10) for _ in range(2)] + [50 + np.random.randint(-10, 10)],
            [50 + np.random.randint(-10, 10)] + [200 + np.random.randint(-10, 10) for _ in range(2)] + [50 + np.random.randint(-10, 10)],
            [50 + np.random.randint(-10, 10) for _ in range(4)]
        ]
        X_cat_mock.append(cat_variation)

        # Slight variation for DOG
        dog_variation = [[150 + np.random.randint(-10, 10) for _ in range(4)] for _ in range(4)]
        X_dog_mock.append(dog_variation)
    return X_cat_mock, X_dog_mock    