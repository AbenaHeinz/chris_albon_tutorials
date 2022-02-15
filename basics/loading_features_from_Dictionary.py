from typing import Dict
from sklearn.feature_extraction import DictVectorizer

# create a dictionary 
staff = {"name": "Steve Miller", "age": 33,
"name": "Lyndon Jones", "age": 12,
"name": "Baxter Morth", "age": 18}

# Convert Dictionary To Feature Matrix

# Create an object for our dictionary vectorizer

vec = DictVectorizer()

# Fit then tranform the staff dictionary with vec, the output the array
vec.fit_transform(staff).toarray()

# Get feature names

vec.get_feature_names()

