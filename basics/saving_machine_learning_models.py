from cmath import pi
from random import random
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pickle
from sklearn.externals import joblib

# load data 

'''load the iris data '''
iris = datasets.load_iris()

'''create a matrix X, of features and a vectors. y'''
X, y = iris.data, iris.target


# Train Model  
'''Train a naive logistic regression model'''
clf = LogisticRegression(random_state=0)
clf.fit(X,y)

# Save to string using Pickle
'''Save the trained modele as a picke string '''
saved_model = pickle.dumps(clf)

'''View the Pickled Model'''
saved_model

'''load the Pickled Model'''
clf_from_pickle = pickle.loads(saved_model)

'''Use the loaded pickled model to make predictions'''

clf_from_pickle.predict(X)

# Saved to Pickled file usinhg joblib

'''Save the modle as a pickle in a file '''
joblib.dump(clf, "filename.pkl")

'''Load the model from the file'''
clf_from_joblib = joblib.load("filename.pkl")

'''Use the loaded model to make predictions '''
clf_from_joblib.predict(X)