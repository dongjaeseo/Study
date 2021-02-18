# import p31_sample
from p31_sample import test

x = 222

def main_func():
    print('x : ', x)

main_func()

test()
# 이 파일의 x와 다른 파일의 x는 다른 메모리에 할당되어있다!