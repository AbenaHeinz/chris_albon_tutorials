'''
Reciever Operating Characteristics(ROC)

Shows the true position and flas positive raste for ever probablility threshold for a binary classifer
'''

#load libraries

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# generate feature and target vectors  
X, y = make_classification(n_samples = 10000, n_features = 10, n_classes = 2, n_informative = 3, random_state = 3)

#Split into training and test sets  
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1, random_state=1)

#create classfier
clf = LogisticRegression()

#train model  
clf.fit(X_train, y_train)

# get predicted probabilitues  
y_score = clf.predict_probra(X_test)[:,1]

# create thure and false postive rates 
false_positive_rate, true_postive_rates, threshold = roc_curve(y_test, y_score)

# Plot ROC Curve 
plt.title("Reciever Operating Curve")
plt.plot(false_positive_rate, true_postive_rate)
plt.plot([0,1], ls = "--")
plt.plot([0,0], [1,0], c= " .7"), plt.plot([1, 1], c= " .7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Postive Rate")
plt.show() 

