# load libaries  
import string
import numpy as np

# create text 
text_data = ["Hi!!!! I. Love. This. Song.... ", "10000% Agree!!!! #LoveIT", "Right?!?!"]

# create function sing string.punctuatuon to remove all puntucation  
def remove_punctuation(sentence: str) -> str:
    return sentence.translate(str.maketrans('','',string.punctuation))


# Apply Function

[remove_punctuation(sentence) for sentence in text_data]