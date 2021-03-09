# 즉시 실행 모드
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

# print(tf.executing_eagerly()) # False # True 2.3.1 버전에서는 디폴트가 트루!

# 텐서플로 2점대에서 1점대의 코딩이 가능하다
tf.compat.v1.disable_eager_execution()

# print(tf.executing_eagerly()) # False # False 위 코드를 실행하면 펄스가 된다
# 즉 텐서플로 2 버전에서도 1처럼 코딩할 수 있다


print(tf.__version__)

hello = tf.constant('Hello World')
print(hello)
# 텐서플로는 자료형이 세개가 있는데 / 상수, 변수, 플레이스홀더?(입력만 받)
# 그냥 실행시키면 자료형만 출력이 된다
# Tensor("Const:0", shape=(), dtype=string)

# 그래서 세션을 만들어 실행시켜야 출력이 된다
# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello)) # b'Hello World'


