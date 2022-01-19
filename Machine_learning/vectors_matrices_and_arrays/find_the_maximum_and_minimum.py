# load libaries 
import numpy as np

# create matrix 
matrix = np.array([[1,2,3], [4,5,6], [7,8,9]])

# find the maximum element
np.max(matrix)

# find the minimum element 
np.min(matrix)

# finding the maximum element in each column
np.max(matrix, axis=0)
# finding the minimum element in each row
np.max(matrix,axis=1)