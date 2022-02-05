'''
Outlier

Drop - Not a great option. We lose lots of information. Find out if genuine extreme value of broken sensor. 
Mark - Safest option. We can see if the outlier has and effect 
Rescale - Log Values so outlire dont have as great an effect 
'''

# Import Libaries  
import pandas as pd

# Create Dataframe  

houses = pd.DataFrame()
houses["Prices"] = [534433, 392333, 293222, 4322032]
houses["Bathrooms"] = [2, 3.5, 2, 116]
houses["Square_Feet"] = [1500, 2500, 1500, 48000]

houses

# Option 1 - Drop 
# Drop obsercations greater than some value  
houses[houses["Bathrooms"]<20]

# Option 2 - Mark  
# load libaries  
import numpy as np

# create feature based on boolean condition
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)

#show data 
houses

# Option 3 - Rescale  

# Log Features 
houses["Log_of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]

# Show Data 
houses


