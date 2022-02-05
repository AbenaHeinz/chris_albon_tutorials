#load library

import numpy as np

# create matrix
matrix = np.array([[1,2,3], [4,5,6], [7,8,9]])

# create a vectorized function that adds 100 to something 
add_100 = lambda i: i+100

# create a vectorized function
vectorized_add_100 = np.vectorize(add_100)

# apply functions to elements 
vectorized_add_100(matrix)
