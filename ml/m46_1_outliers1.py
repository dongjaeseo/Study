# 이상치 처리
# 1. 0 처리
# 2. Nan 처리 후 보간
# 3, 4, 5, 알아서 해

import numpy as np

aaa = np.array([1,2,3,4,6,7,90,100,500,1000])

def outliers(data_out):
    quantile_1, q2, quantile_3 = np.percentile(data_out, [25, 50, 75])
    print('1 사분위 : ', quantile_1)
    print('q2  : ', q2)
    print('3 사분위 : ', quantile_3)
    iqr = quantile_3 - quantile_1
    lower_bound = quantile_1 - (iqr*1.5)
    upper_bound = quantile_3 + (iqr*1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outlier_loc = outliers(aaa)
print('이상치의 위치 : ', outlier_loc)
# 이상치의 위치 :  (array([8, 9], dtype=int64),)

import matplotlib.pyplot as plt

plt.boxplot(aaa)
plt.show()