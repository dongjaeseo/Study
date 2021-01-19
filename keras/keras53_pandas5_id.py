import pandas as pd
df = pd.DataFrame([[1,2,3,4],[4,5,6,7],[8,9,10,11]],columns = list('abcd'), index = ('가','나','다'))
print(df)

df2 = df  ## 같은 메모리를 공유해서 새로운게 만들어지는게 아님 >  = 만 해당된다

df2['a'] = 100

print(df2)
print(df)

print(id(df), id(df2))

df3 = df.copy() ## 새로운 데이터프레임을 만드는것임

df2['b'] = 333

print(df)
print(df2)
print(df3)

df = df + 99 ## = 이 아니고 사칙연산이 들어가면 새로운 데이터프레임을 생성한다                                               
print(df)
print(df2)