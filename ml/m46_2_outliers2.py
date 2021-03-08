# 실습
# outliers1 을 행렬형태도 적용할 수 있도록 수정

import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
                [1100,1200,3,1400,1500,1600,1700,8,1900,2000]])
                
print(aaa.shape) # (2, 10)
print('==================================')

def outliers(data_out):
    loc = []
    for row in range(data_out.shape[0]):
        quantile_1, q2, quantile_3 = np.percentile(data_out[row], [25, 50, 75])
        print('1 사분위 : ', quantile_1)
        print('q2  : ', q2)
        print('3 사분위 : ', quantile_3)
        print('==================================')
        iqr = quantile_3 - quantile_1
        lower_bound = quantile_1 - (iqr*1.5)
        upper_bound = quantile_3 + (iqr*1.5)
        loc.append(np.where((data_out[row]>upper_bound) | (data_out[row]<lower_bound)))
    return loc

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

# 이상치 부분에 0 으로 바꿔주기!
loc = outliers(aaa)
for row in range(len(loc)):
    for l in loc[row][0]:
        aaa[row][l] = 0

plt.boxplot(aaa)
plt.show()

