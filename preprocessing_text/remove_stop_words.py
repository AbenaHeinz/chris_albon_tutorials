# load library  
from nltk.corpus import stopwords

# you will have to download the set of words for the first time  
import nltk
nltk.download("stopwords")

# Create work token  
tokenized_words = [ "I", "am", "going", "to", "go", "to", "the", "store", "and", "park"]

#load stop words 
stop_words = stopwords.words('english')

#s Show stop words  
stop_words[:5]

# remove stop words 
[word for word in tokenized_words if word not in stop_words]