# load libraries 
from sklearn import datasets
import matplotlib.pyplot as plt

# Load Digits Dataset 

'''Digits is a dataset of handwritten digits. Each feature is the intensity of one pixel of an 8x8 image'''

# Load digits dataset  
digits = datasets.load_digits()

# create feature matrix  
X = digits.data()

# create target vector
Y = digits.target()

# view the first observation'd feature values

X[0]

'''The observation's feature values as presented as a vector/ However by doing the images method we can load 
the same feature values as a matrix and then visualise the actual handwritten character'''

# Veiw the first observation's feature value as a matrix
digits.images[0]

# Visualise the first obsercation's feature values as an image 

plt.gray()
plt.matshow(digits.images[0])
plt.show()




