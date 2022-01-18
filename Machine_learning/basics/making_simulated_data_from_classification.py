# load libraries 
from sklearn.datasets import make_classification
import pandas as pd

# Create simulated feature matrix and output vectors with 100 samples, 
features, output = make_classification(n_samples = 100, 
                                        #ten features
                                        n_features=10,
                                        #five features that actually predict the output classes 
                                        n_informative=5,
                                        #five features taht are random and unrelated to the output
                                        n_redundant=5,
                                        # three output classes 
                                        n_classes=3, 
                                        # with 20% of observations in the first class, %30 in the second 
                                        # and 50% in the third class "None" making balance classes
                                        weights=[.2,.3,.8])
# View Data  

# View t the first five observations and there 10 features
pd.DataFrame(features).head()

# view the first five observation's classes
pd.DataFrame(output).head()