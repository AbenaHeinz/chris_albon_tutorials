'''
Principal Components Analysis(PCA) is a common feature extraction method in data science. Technically , PCA
finds the eigenvectors of a covariance matrix with the highest eigenvector and then yse those to project  
the fata into a new subspace and equal or less dimensions .  Practically, PCA converts a matrix of n features  
into a new dataset for (hopefully) less than n features. That is, it reduces the number of features by constructing 
a new smaller number variable wheich capture a significant portion of the information found in the orignal features 
However, the goal of this tutorical is no to explain the concept of PCA that is done very well elsewhere but rather to 
demonstrate PCA in action 
'''

#import libraries  

import numpy as np
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler

# load the brest cancer dataset  
dataset = datasets.load_breast_cancer()
# load the  features
X = dataset.data


#view the shape of the dataset  
X.shape

# veiw the data  
X

# create a scalar object  
sc = StandardScaler()

# fit the  scaler to the feature and transform  
X_std = sc.fit_transform(X)

'''
Conduct PCA

Notice that PCA contains a parameer, the number of components. This is the number of output features and will need to be tuned 
'''

# create a pca object with the 2 components as a parameter  
pca = decomposition.PCA(n_components=2)

# fit the PCA and transform the data  
X_std_pca = pca.fit_transform(X_std)

'''
View new features  

After the PCA, the new data has been reduced to two features, with  the same number of rows as the original feature
'''

# View the new feature data's shape
X_std_pca.shape

# View the new feature data  
X_std_pca