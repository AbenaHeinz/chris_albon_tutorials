'''
Mean Imputation replaces missing values with the mean value of that feature/variable. 
Mean Imputation is one of the most "naive" imputation methods because unlike more complex methods like K-nearest neigbours imputation,  
It does not use the information we have about and observation to estimate a value for it  
'''

#load libaries  
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


#create an empty dataset  

df = pd.DataFrame()

# create two variables called x0 and x1, Make teh first value of x1 a missing value  
df["x0"] = [0.3051, 0.4949, 0.3769, 0.2231, 0.341, 0.4436, 0.5897, 0.6308, 0.5]
df["x1"] = [np.nan, 0.2654, 0.2615, 0.5846, 0.4615, 0.8308, 0.4962, 0.3269, 0.6731]

#view dataset
df

# Fit Imputer 
# Create an imputer object that loks for  "NaN" values, then replace them with the mean value of the feature by columns (axis=0)

mean_imputer = SimpleImputer(missing_values = "NaN", strategy = "mean")
#train the imputer on the df dataset  
mean_imputer = mean_imputer.fit(df)

#apply the imputer in the df dataset  
imputed_df = mean_imputer.transform(df.values)

# View the data  
imputed_df


# Apply Imputer  

imputed_df = mean_imputer.transform(df.values)