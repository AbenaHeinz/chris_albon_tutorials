from curses import window
import pandas as pd

# create datetimes

time_index = pd.date_range("01/01/2010", periods=5, freq="M")

#create dataframe set index

df = pd.DataFrame(index=time_index)

# create feature 
df["Stock Prices"] = [1,2,3,4,5]

#calculate rolling mean  

df.rolling(window = 2).mean()

#Identify max value in rolling time window  
df.rolling(window = 2).max()