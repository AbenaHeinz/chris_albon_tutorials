'''
if the features  are catergorical , calculate a chi-square(X**2), statistic between each feature  and the target vector 
However, if the features are qunatative, compute the ANOVA F-value betwee each feature and the target vector  

The F-value scares examine if, when we group the numerica features by the larger vector. the means of eacg group
are significantly different
'''

# load libraries  
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# load iris data 
iris = load_iris()
# create features and targets  
X = iris.data
y = iris.target

# create a SelectKBest object to select  features  with two best ANOVA F-values  
fvalue_selector = SelectKBest(f_classif, k=2)

# apply the SelectKBest object to the  features and targets
X_kbest = fvalue_selector.fit_transform(X, y)

# show results  
print("Original number of features:", X.shape[1])
print("Reduced number of features:", X_kbest.shape[1])
