# load libraries

from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Create numpy array  

y =[("Texas", "Florida"), ("California", "Alabama"), ("Texas", "Florida"), ("Delware", "Florida"), ("Texas", "Alabama")]

# create MultiLabelBinarizer

one_hot = MultiLabelBinarizer()

# one hot encode data

one_hot.fit_transform(y)

# View Classes 

one_hot.classes_
