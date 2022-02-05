# oad libaries  

from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

#load data 
iris = load_iris()
#create target vector and feature matrix
X,y = iris.data, iris.target

# split into trianing and test stes  
X_train,X_test, y_train, y_test = train_test_split(X,y, random_state=0)

# create dummy classifier  
dummy = DummyClassifier(strategy = "uniform", random_state = 1)
# train model  
dummy.fit(X_train,y_train)

# get accuracy score
dummy.score(X_test, y_test)
