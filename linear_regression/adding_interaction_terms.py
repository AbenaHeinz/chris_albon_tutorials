'''
Interaction Terms  

Interaction terms allow us model relationships when the effects of a feature on the target is influenced by another feature 
'''

# load libraries 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
import warnings
# supress warning 
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# load the data with only two features  
boston = load_boston()
X = boston.data[:,0:2]
y = boston.target

'''
Add Interation Term  

Interaction effects can be accounted for by including a new feature comprising the products of coressponding calies from the inteaction features:

$$\hat_{y} = \hat_{{\beta_{0}} + \hat_{\beta{1}}x_{1}+\hat_{\beta{2}x_{2} + \hat_{\beta_{3}}x_{1}x_{2} + \epsilon} $$

where x1 and x2 are the values of the two features. respecticely and x1x2 represent the interacttion between the two.
It can be useful to use scikit-learn's PolyomialFeatures to creative interative terms for all combinaton features.
We can then use model selection strategies to identify the combination of features and interaction term  which produce the best model
'''


# Create interaction term (not polynomioal features)
interaction = PolynomialFeatures(degree=3, include_bias = False, interaction_only=True)
X_inter = interaction.fit_transform(X)

# Create liner regression 
regr = LinearRegression()

# fit the linear regression 
model = regr.fit(X_inter, y)