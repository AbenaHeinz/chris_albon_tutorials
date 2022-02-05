from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load data 
iris = datasets.load_iris()
#create feature matrix  
X = iris.data
#create target
y = iris.target
# create list of target class names  
class_names = iris.target_names

# create training and test set  
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state = 1)

# create a logistics regression  
classifier = LogisticRegression()

# train model and make predictions  
y_hat = classifier.fit(X_train, y_train).predict(X_test)

#generate a classfication report  
print(classification_report(y_test, y_hat, target_names=class_names))

# note - suport referes to the number of observations in each class