# load library  

import pandas as pd

# create dataframe
df = pd.DataFrame()

# create five dates 
df["date"] = pd.date_range("1/1/2001", periods=150, freq="w")

# create features for year, month, day, hours and minutes

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["hour"] = df["date"].dt.hour
df["minute"] = df["date"].dt.minute

# show first three rows
df.head(3)