import numpy as np
import pandas as pd

# create string  
date_strings = np.array(["03-04-2005 11:35PM", "23-05-2010 12:01AM", "04-09-2009 09:09PM"])

# convert string into timestamp
[pd.to_datetime(date, format = "%d-%m-%Y %I:%M %p", errors="coerce") for date in date_strings]