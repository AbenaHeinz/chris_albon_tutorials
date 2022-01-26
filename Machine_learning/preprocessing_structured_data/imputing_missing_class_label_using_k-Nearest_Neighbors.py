import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# create feature matrix with catergorical features  

X = np.array([[0, 2.10, 1.45],[1, 1.18, 1.33],[0, 1.22, 1.27],[1, -0.21, -1.19]])

# create feture matrix with missing values  

X_with_nan = np.array([[np.nan, 0.87, 1.31], [np.nan, -0.67, -0.22]])

#Train KNN Learner

clf = KNeighborsClassifier(3, weights="distance")
trained_model = clf.fit(X[:, 1:], X[:, 0])

# predict missing values class

imputed_values = trained_model.predict(X_with_nan[:,1:])

# join column of predicted classes with the other features  
X_with_imputed = np.hstack((imputed_values.reshape(-1, 1), X_with_nan[:,1:]))

# join two features matrices 

np.vstack((X_with_imputed, X))