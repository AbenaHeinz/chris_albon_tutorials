'''
Chi-cquared for featured selectuon  

To Use X**3 for feature selecton, we calculate x**3 between each feature and the target and select the desired
number  of features with the best x**3 score

The intution is that is a feature is independent to the target, it is uniformative for classifying observations
'''

#load libraries  
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#load iris data
iris = load_iris()
#create features and target  
X = iris.data
y = iris.target

# convert to cataragorical data by converting data to intergers  
X = X.astype(int)

#Select two feature with highest chi-squared statistics  
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X,y)

# show results  
print("Original Number of Features:", X.shape[1])
print("Reduced Number of Features:", X_kbest.shape[1] )