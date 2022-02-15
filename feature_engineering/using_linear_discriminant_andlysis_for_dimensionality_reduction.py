# load linaries  
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# load iris data  
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create an LDA that will reduce the data down to 1 feature 
lda = LinearDiscriminantAnalysis(n_components=1)
# run an LDA and use it to transform the feature  
x_lda = lda.fit(X,y). transform(X)

# print the number of features
print("Original number of features", X.shape[1])
print("Reduce number of features:", x_lda.shape[1])

# view the ratio  of explained variance  

lda.explained_variance_ratio_
