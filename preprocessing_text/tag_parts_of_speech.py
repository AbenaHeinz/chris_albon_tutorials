from nltk import pos_tag, word_tokenize

# create text data

text_data = "Chris loved outdoor running"

# tag parts of speech  

# use pre-trained parts of speech tagger  
text_tagged = pos_tag(word_tokenize(text_data))

text_tagged