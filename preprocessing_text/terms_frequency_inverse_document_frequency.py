'''
TF-IDF - Terms Frequency Inverse Document Frequency 
TF-IDF is a measure of originality  of a word by comparing he number of times a word appears in a doc 
with the number of docs in a words appear in a dock with the number of docks the words appear
'''

import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# create text data  
text_data = np.array(["I love Brazil, Brazil", "Sweden is best", "Germany beats both"])

#create the tf-idf feture matrix
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

# show tf-idf feature matrix
feature_matrix.toarray()

#show tf-idf feature matrix
tfidf.get_feature_names()

# create a dataframe 
pd.DataFrame(feature_matrix.toarray(), columns=tfidf.get_feature_names())