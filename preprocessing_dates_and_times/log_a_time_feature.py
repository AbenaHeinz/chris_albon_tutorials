import pandas as pd

# create data frame
df = pd.DataFrame()

# create data  
df["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
df["stock_prices"] = [1.1,2.2,3.3,4.4,5.5]

#log values by one row
df["previous_day_stock_price"] = df["stock_prices"].shift(1)

#show dataframe  

df