import pandas as pd

#create  dates  
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))

#view dates  
dates

# show days of the week 
dates.dt.weekday_name