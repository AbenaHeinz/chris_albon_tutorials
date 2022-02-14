# load library  
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# create feature matrix  
X = np.array([[2,3], [2,3], [2,3]])

# create PolynomialFeature object with intweraction_only set to True
interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Transform Feature matrix
interaction.fit_transform(X)