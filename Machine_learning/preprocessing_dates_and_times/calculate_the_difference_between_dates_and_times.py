import pandas as pd

#create dataframe 

df = pd.DataFrame()

# create two datetime features  
df["Arrived"] = [pd.Timestamp("01-01-2017"), pd.Timestamp("01-04-2017")]
df["Left"] =  [pd.Timestamp("01-01-2017"), pd.Timestamp("01-06-2017")]

# calculate difference(Method1)
# calculate the difference between teh features  
df["Left"] - df["Arrived"]

# calculate difference(Method2)
# calulate the difference between teh features  
pd.Series(delta.days for delta in (df["Left"] - df["Arrived"]))