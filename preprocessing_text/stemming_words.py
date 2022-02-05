'''
Stemming reduces a word to its stem. The result is less readable by humans byt makes the text there comparable across observation
Example = "Tradition" and "Traditionial" have the same stem "tradit"
'''

# Load Libaraies  
from nltk.stem.porter import PorterStemmer

# create word tokens  

tokenized_words = ["I", "am", "humbled", "by", "this", "traditional", "meeting"]

'''Stemming reduces a word to its stem by identifying and removing affixes(e.g gerunds) while keeping the root
meaning of the word.NTLK's ```PorterStemmer``` implements the widely used Porter stemming algorithms'''

# create stemmer
porter = PorterStemmer()

# apply stemmer
[porter.stem(word) for word in tokenized_words]
