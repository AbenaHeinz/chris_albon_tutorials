# load libaries  

from audioop import minmax
from xml.sax.handler import feature_external_pes
from sklearn import preprocessing
import numpy as np

# create feature
x = np.array([[-500.5], [-100.1], [0], [100.1],[900.9]])

# rescale Features using Min-Max

# create a scaler  
minmax_scale =  preprocessing.MinMaxScaler(feature_range=(0,1))

# Scale Feature  
x_scale = minmax_scale.fit_transform(x)

# Show Feature  

x_scale
