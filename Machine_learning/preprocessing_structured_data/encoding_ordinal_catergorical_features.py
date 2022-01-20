#load libraries 
import pandas as pd



# create features  
df = pd.DataFrame({"Score" : ["Low", "Low", "Medium", "Medium", "High"]})
#view dataframe 
df
#create mapper

scale_mapper = {"Low":1,"Medium":2, "High":3}
#map features values to scale
df["Scale"] = df["Score"].replace(scale_mapper)

df