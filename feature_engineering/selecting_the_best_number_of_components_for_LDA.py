'''
In scikit-learn LDA is implimented using LinearDiscriminatAnalysis includes a parametere n-components indicating
the number of features we want returned,  To figure out  what argument values to use with n_components (e.g. 
how many parameters to keep). we can  take advantage of the fact that explained_variance_ration_ tells use the variance 
explained by each outputtered features and is a sorted array  

Specifically we can run LinearDiscriminantAnalysis with n_components set to None to reurn ratio of variance explained by
every components feature, then calculated how many components are required to get above same threshold of variance explained
(often 0.95 or 0.99)
'''

#Load libararies 
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# load iris data  
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create and run an LDA
lda = LinearDiscriminantAnalysis(n_components=None)
x_lda = lda.fit(X, y)

# create array of explained variance ratio
lda_var_ratios = lda.explained_variance_ratio_

# create a function  
def select_n_components(var_ratio, goal_var: float) -> int:
    #set intiial variance explained so far
    total_variance = 0.0
    #Set initial number of features 
    n_components = 0
    # for the explained variance for each feature, add the explain variance of each feature  
    # add one to the number of components , if we reach our goal level of expplained variances, end the loop
    # return the number of components  

    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break
    return n_components
# run functions 
select_n_components(lda_var_ratios, 0.85)