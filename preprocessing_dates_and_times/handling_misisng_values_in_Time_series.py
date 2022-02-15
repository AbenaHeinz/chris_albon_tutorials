import pandas as pd
import numpy as np


# create date

time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# create dataframe set index

df = pd.DataFrame(index=time_index)

#ceate features with a gap of isisng objects  

df["Sales"] = [1.0, 2.0, np.nan, np.nan, 0.5]

#interpolate missing values  
df.interpolate()

#forward fill misisng values  
df.ffill()

#back-fill
df.bfill()

#Interpolate missing values  
df.interpolate(limit=1, limit_direction="forward")
