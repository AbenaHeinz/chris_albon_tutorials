import pandas as pd
import numpy as np

# create feature matrics with two highly correlated features  
X = np.array([[1,1,1], [2,2,0], [3,3,1], [4,4,0], [5,5,1], [6,6,0], [7,7,1], [8,7,0], [9,7,1]])

#convert feature matrix into DataFrame  
df = pd.DataFrame(X)

# view the data frame  
df

# Identify highly correlated features  

# create correlation matrix  
corr_matrix = df.corr().abs()

# select the upper triangle of corellation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

#Find index of feature columns with corellation gratea than 0.95
to_drop = [column for column in upper.columns if any (upper[column]> 0.95)]

# drop marked features  
df.drop(df[to_drop], axis = 1)