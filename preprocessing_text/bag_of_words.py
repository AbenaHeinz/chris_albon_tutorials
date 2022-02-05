'''
Bag of Words  

Converts text to a matrics where every row is an observation and every feature is a unique word.  
The Value of each element in the matrix is either a binary indicator marking the precents of the word or 
an integer of the number of times that word appears
'''

# Load Libaries 

from matplotlib.pyplot import text
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# create text  

text_data = np.array(["I love Brazi, Brazil", "Sweden is Best", "Germany beats both"])

# create the bad of words feature matrix  
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)


# Show feature matrics  
bag_of_words.toarray()

# get features name 
feature_names = count.get_feature_names()
#view features names  
feature_names


# create a data frame  
pd.DataFrame(bag_of_words.toarray(), columns=feature_names)
