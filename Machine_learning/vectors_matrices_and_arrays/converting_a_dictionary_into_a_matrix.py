# import library 
from sklearn.feature_extraction import DictVectorizer

#create a dictionary of data 

data_dict = [{ "Red": 2, "Blue": 4},{"Red": 4, "Blue" : 3}, {"Red": 1, "Yellow": 2}, {"Red": 2, "Yellow": 2}]

#create Dicvectorizer object 
dictVectorizer = DictVectorizer(sparse=False)

# convert dictoionary into feature matrix
features = dictVectorizer.fit_transform(data_dict)

#view feature matrix
features

# View column names  
# view feature matrix column names

dictVectorizer.get_feature_names()