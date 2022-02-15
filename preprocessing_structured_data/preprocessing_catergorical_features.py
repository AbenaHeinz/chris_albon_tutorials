from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import pandas as pd

# create data 
raw_data = {"first_name" : ["Jason", "Molly", "Tina", "Jake", "Amy"], "last_name":["Miller", "Jacobson", "Ali", "Milner", "Cooze"], "age":[42, 52, 36, 24, 73], "city":["San Franciso", "Baltimore", "Miami", "Douglas", "Boston"]}
df = pd.DataFrame(raw_data, columns=["first_name", "last_name", "age", "city"])
df

# Convert Nominal Catergorical Features into Dummy Variables Using Pandas 
# Create dummy varibales for ever unique catergory in df.city  

pd.get_dummies(df["city"])

# convert Nominal Catergorical Data into Dummy (OneHot) features using Scikit 

# Convert strings catergorical names in integers 

integerized_data = preprocessing.LabelEncoder().fit_transform(df["city"])

# view data  

integerized_data

# convert interger catergorical representation to OneHot encoding s

preprocessing.OneHotEncoder().fit_transform(integerized_data.reshape(-1,1)).toarray()