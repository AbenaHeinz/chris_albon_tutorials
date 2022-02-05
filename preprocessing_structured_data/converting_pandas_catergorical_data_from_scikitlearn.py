# import required packages 

from sklearn import preprocessing
import pandas as pd

# create dataframe 

raw_data = {"Patient":[1,1,1,2,2], "obs":[1,2,3,1,2], "treaatment":[0,1,0,1,0], "score":["strong", "weak", "normal", "weak", "strong"]}
df= pd.DataFrame(raw_data, columns=["Patient", "obs", "treatment", "score"])

#fit the lable encoder  
# create a label (catergory) encoder object
le = preprocessing.LabelEncoder()

#fit the endcoder for the pandas column
le.fit(df["score"])

#View the label  
#View the label  

le.transform(df["score"])

#transform intergers into catergories 
#convert some intergers into their catergory names 
list(le.inverse_transform([2, 2, 1]))

