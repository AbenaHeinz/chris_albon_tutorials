import numpy as np
import pandas as pd

# Drop missing values using Numpy  
#create featured matrix

X = np.array([[1,2],[6,3],[8,4], [9,5], [np.nan,4]])

# remove observations with missing values using numpy 
X[~np.isnan(X).any(axis=1)]

# Drop missing values using pandas
# load data as a datafram  
df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])

#remove observations with missing values 
df.dropna()
