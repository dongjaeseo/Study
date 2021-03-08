import numpy as np
from sklearn.covariance import EllipticEnvelope

aaa = np.array([[1,2,-10000,3,4,6,7,90,100,500,1000],
                [10000,11000,12000,-100000000000,14000,15000,16000,17000,8,19000,20000]])
aaa = np.transpose(aaa)
print(aaa.shape)

outlier = EllipticEnvelope(contamination=.05)
outlier.fit(aaa)

print(outlier.predict(aaa))