from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['3/1/2021', '3/2/2021', '3/3/2021', '3/4/2021', '3/5/2021']
dates = pd.to_datetime(datestrs)
print(dates) # DatetimeIndex(['2021-03-01', '2021-03-02', '2021-03-03', '2021-03-04','2021-03-05'], dtype='datetime64[ns]', freq=None)
print("==========================================")

ts = Series([1, np.nan, np.nan, 8, 10], index = dates)
print(ts)

ts_intp_linear = ts.interpolate() # bogan 판다스에서 제공, 가급적이면 시계열 데이터
print(ts_intp_linear)