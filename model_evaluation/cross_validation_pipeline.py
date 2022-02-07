'''
The code below does a lot in only a fewlindes to help explain things;here are the steps that code is doing 

1. Split the raw data into three foldes, select one for testing and two for training  
2. Preprocess the data by scaling the training features.  
3. Train a support vector classifier on the training data
4. Apply the classifire to the test data 
5. Record the accuracy score 
6. Repeat step 1-5 two more times, once for each fold
7. Calcualte the mean score for each fold  
'''

from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm


''' Load Data 
For this tutorial we will use the famous iris dataset. This iris data contains four measurements of 150 iris flowers
and there species. We will use a support vector classfier to predict the speices of the iris flowers
'''

# load the iris test data  
iris = load_iris()

# view the iris data features for the first three rows 
iris.data[0:3]

# view the iris data targets for first three rows. '0' means it flower is of the setosa species

iris.target[0:3]

'''
Create Classifier Pipeline

Now we create a pipeline for the data. first the pipeline preprocessed the data by scaling the features variables's
value to mean zero and unit variance. Second, the pipline trains a support classifier of the data with C-1. C
is the cost function for the margins. The higher the C. Th elest tolerant the nidek is for the observations 
being on the wrong side of the hyperplane
'''

# Create a pipeline that scales the data then traines a support vector classifier

classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))

'''
Cross Validation 

Scikit provides a great helper function to make it easy to do cross validation. 
Specifically the cost below splits the data into three folds, then executes the classifier pipeline to the iris data. 

Imoortant note from the scikit docs: for interger/None inputs, if y is binairy or multiclass, StratifiedKFold used. 
if the estimatore is a classfier or if y is neither binary nor multiclassed. KFold is used
'''

# Kfold/StratifedKfolds cross validation with three folds (the default)
# applying the classifier pipeline to the feature and target data  
score = train_test_split.cross_val_score(classifier_pipeline, iris.data, iris.target, cv = 3)

