class Person:
    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address
    
    def greeting(self):
        print('안녕하세요, 저는 {0}입니다.'.format(self.name))
    
    def intro(self):
        print(f'제 나이는 {self.age}살이고 주소는 {self.address}입니다!')

# 클래스 안에 있는건 기능과 내용? 변수 함수 짬뽕으로 다 넣엉
# 클래스 이닛은 초기화?