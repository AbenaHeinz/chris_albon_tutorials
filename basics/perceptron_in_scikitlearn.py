from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# load iris data 
iris = datasets.load_iris()
#create our X and y dataset
X = iris.data
y = iris.target

# View the iris data 

'''View the first five observations of our y data'''
y[:5]

'''View thw first 5 observations of our X data, notice that there are fore independent variable features'''
X[:5]

# Split the iris data into Training and Testing 
'''Split the data into 70% training data and 30% testing data'''
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

# Preprocess the X data by scaling 

'''Train the scaler, which standardizes all the features to have mean=0 and unit variance'''
sc = StandardScaler()
sc.fit(X_train)
'''Apply the scalar to the X training data'''
X_train_std = sc.transform(X_train)
'''Apply the same scala to the X test data '''
X_test_std = sc.transform(X_test)

# Train a Perceptron Learner  

'''Create a perceptron object with the parameters: 40 iterations(epochs) over the data an a learn'''
ppn = Perceptron(max_iter = 40, eta0 = 0.1, random_state = 0)

'''Train the preceptron'''
ppn.fit(X_train_std, y_train)

# Apply the Trained Learner to Test Data 
'''Apply the trained perceptron on the X data to make prediction for the test data'''
y_pred = ppn.predict(X_test_std)

# Compare the Predicted Y with the True Y 

''' View the predicted y test data'''
y_pred
'''View the true test data '''
y_test

# Examine Accuracy Matric

'''Veiw the accuracy of the model which is : 1 - (observations predicted wrongs/total wrongs)'''
print("Accuracy: %.2f"% accuracy_score(y_test, y_pred))