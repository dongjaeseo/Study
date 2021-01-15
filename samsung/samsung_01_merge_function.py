import numpy as np
import pandas as pd

def newdf(new,old,days):
    df_new = pd.read_csv(open(new), index_col = None, header = 0, encoding = 'cp949', thousands = ',')
    df_new = df_new.loc[:,['일자','시가','고가','저가','종가','거래량','금액(백만)']]
    df_new = df_new.iloc[:days, :]

    df_old = pd.read_csv(open(old), index_col = None, header = 0, encoding = 'cp949', thousands = ',')
    df_old = df_old.loc[:,['일자','시가','고가','저가','종가','거래량','금액(백만)']]

    if df_old.iloc[0,0] != '2021/01/15':
        df = df_new.append(df_old, ignore_index = True)
        df.to_csv('./samsung/삼성전자_병합.csv', sep =',', encoding = 'cp949',index = False)
        return(df)
    else:
        return(df_old)

days = 1
df = newdf('./samsung/삼성전자2.csv','./samsung/삼성전자.csv',days)
df = newdf('./samsung/삼성전자0115.csv','./samsung/삼성전자_병합.csv',days)

# 1. 삼성전자병합 파일에 삼성2 첫줄+ 삼성 을 붙여준다
# 2. 삼성전자병합 파일에 삼성0115 첫줄 + 삼성병합 을 붙이고 덮어 씌운다

df_old = pd.read_csv('./samsung/삼성전자_병합.csv', index_col = None, header = 0, encoding = 'cp949', thousands = ',')

