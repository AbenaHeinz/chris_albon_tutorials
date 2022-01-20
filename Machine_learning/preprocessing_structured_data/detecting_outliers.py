#load libaries 
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs


# Create simulated data  
X, _ = make_blobs(n_samples=10, n_features=2, centers=1, random_state=1)
#replace the first observations's value  with extreme values 

X[0,0] = 10000
X[0,1] = 10000

# detect the outliers 
'''```ElipticalEnvelope``` assumes the data is normally distrubited bases on thhat assumption "draws" an ellipse
around the data classifying any observation inside the ellipsisasan inlier(labled as 1) and any observation outside
the elipsise as an outlier (labeled as -1). A major limitation of this approach is the need to specify a ```contamination```
parameter which is the proportioncal of observations that Sthe outliers, a value that we dont know'''

# create detector  
outlier_detector = EllipticEnvelope(contamination=.1)

# fit detector
outlier_detector.fit(X)

#predict outlier
outlier_detector.predict(X)