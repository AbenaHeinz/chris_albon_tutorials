from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# load the digits dataset
digits = datasets.load_digits()

# create  the feature matrixs
X = digits.data

# create the target vector
y = digits.target


# create training and test sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.1, random_state=1)

# Create standardizer 
standardizer = StandardScaler()

#fit standardizer to trianing set  
standardizer.fit =(X_train)

# Apply to both training and test sets  
X_train_std = standardizer.transform(X_train)
X_test_std = standardizer.transform(X_test)
 