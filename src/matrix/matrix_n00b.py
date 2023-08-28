import unittest


class MatrixNoob:
    '''
    Working with matrices
    '''

    def is_matrix(self, matrix):
        """
        Check if the matrix is a list of lists and all sublists have the same length
        """
        return isinstance(matrix, list) and all(isinstance(row, list) for row in matrix) and all(len(row) == len(matrix[0]) for row in matrix)

    def list_to_matrix(self, lst: list, n: int):
        """
        Convert a list into a matrix with n columns.

        Args:
            lst (list): The list to be converted into a matrix.
            n (int): The number of columns in the desired matrix.

        Returns:
            matrix
            (list of lists): The resulting matrix where each row is a sublist of n elements from lst.
        """
        matrix = []
        for i in range(0, len(lst), n):
            matrix.append(lst[i:i+n])
        return matrix

    def matrices_multiplication(self, mat1, mat2):
        """
        Example:
        self.m1 = [[2, 2, 2]]

        self.m2 = [[1, 1, 2], 
                   [1, 1, 1],
                   [0, 0, 0]]

        Step 1: We take the first (and only) row of Matrix 1 (m1):
        [2, 2, 2]

        Step 2: We take the first column of Matrix 2 (m2):
          [1,
           1, 
           0] 

        Step 3: Multiply each corresponding pair of numbers, and then add the results:
        (2*1) + (2*1) + (2*0) = 2 + 2 + 0 = 4

        Step 4: Repeat steps for each column of the (m2):
        [[4, 4, 6]]

        """

        # Check if params are matrices
        if not self.is_matrix(mat1):
            raise ValueError("mat1 is not a martix")

        if not self.is_matrix(mat2):
            raise ValueError("mat2 is not a martix")

        # Get and print the shapes of the matrices
        shape_mat1 = (len(mat1), len(mat1[0]))
        shape_mat2 = (len(mat2), len(mat2[0]))
        print(f'Shape of first matrix: {shape_mat1}')
        print(f'Shape of second matrix: {shape_mat2}')

        # Rest of the code (unchanged from the previous version)
        # Check if number of columns in first matrix equals number of rows in second matrix
        if shape_mat1[1] != shape_mat2[0]:
            raise ValueError(
                "The number of columns in the first matrix must be equal the number of rows in the second matrix.")

        # Define the result list
        result = []
        # For each row in the first matrix
        for i_row in range(len(mat1)):
            # Create a new row in the result
            new_row = []
            # Iterate through columns of the second matrix
            for j_col in range(len(mat2[0])):

                dot_product = 0
                for k in range(len(mat1[0])):
                    dot_product += float(mat1[i_row][k]) * float(mat2[k][j_col])
                # Append the new row to the result
                new_row.append(dot_product)

            result.append(new_row)

        # Print the result of the multiplication
        self.print_matrix(result)
        return result

    def transpose(self, matrix):
        # Check if params are matrices
        if not self.is_matrix(matrix):
            raise ValueError("Transpose operation is available for martix only")
        
        print("matrix before T:")
        self.print_matrix(matrix)

        transposed = []
        for i in range(len(matrix[0])):
            new_row = []
            for row in matrix:
                new_row.append(row[i])
            transposed.append(new_row)

        print("matrix after Transpose:")
        self.print_matrix(transposed)

        return transposed

    def print_matrix(self, matrix):
        print("[")
        for row in matrix:
            # Format each number in the row to have 3 decimal places
            formatted_row = [f"{num:.3f}" for num in row]
            print("    " + str(formatted_row) + ",")
        print("]")


    


# TESTS
class NeuralNetworkBasicTest(unittest.TestCase):
    def setUp(self):
        self.matrix = MatrixNoob()
        # Test data for matrices processing
        self.m1 = [[2, 2, 2]]
        self.m2 = [[1, 1, 2],
                   [1, 1, 1],
                   [0, 0, 0]]

    def test_matrices_multiplication(self):
        res = self.matrix.matrices_multiplication(self.m1, self.m2)
        self.assertEqual(res, [[4, 4, 6]])

    def test_matrices_multiplication_different_m_size(self):
        with self.assertRaises(ValueError):
            self.matrix.matrices_multiplication([[1, 2]], self.m2)

    def test_transpose(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

        matrix_transposed = [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]
        ]

        res = self.matrix.transpose(matrix)
        self.assertEqual(res, matrix_transposed)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(NeuralNetworkBasicTest('test_matrices_multiplication'))
    suite.addTest(NeuralNetworkBasicTest('test_matrices_multiplication_different_m_size'))

    suite.addTest(NeuralNetworkBasicTest('test_transpose'))

    runner = unittest.TextTestRunner()
    runner.run(suite)