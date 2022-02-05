import pandas as pd

# create a dataframe 
df = pd.DataFrame()

# create datetimes 
df["date"] = pd.date_range("1/1/2001", periods = 100000, freq = "H")

#Time Range -Method 1 - select observations between the two datetimes  
df[(df["date"] > "2002-1-1 01:00:00") & (df["date"] <= "2002-1-1 04:00:00")]

#Time Range - Method 2

# set index
df = df.set_index(df["date"])

# select observations between two datetimes  

df.loc["2002-1-1 01:00:00": "2002-1-1 04:00:00"]