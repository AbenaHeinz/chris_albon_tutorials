from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data 
boston = load_boston()

# create features  
X, y = boston.data, boston.target

# make test and training split  
X_train, X_test, y_train,y_test = train_test_split(X, y, random_state=0)

#creae a dummy regressor
dummy_mean = DummyRegressor(strategy="mean")
# "train" dummy regressor
dummy_mean.fit(X_train, y_train)

# create a dummy regressor that predicts a constant value  
dummy_constant = DummyRegressor(strategy = "constant", constant=20)

#"train" Dummy regressor 
dummy_constant.fit(X_train, y_train)

#get r-Squareed score
dummy_constant.score(X_test,y_test)
