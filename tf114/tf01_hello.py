import tensorflow as tf
print(tf.__version__)

hello = tf.constant('Hello World')
print(hello)
# 텐서플로는 자료형이 세개가 있는데 / 상수, 변수, 플레이스홀더?(입력만 받)
# 그냥 실행시키면 자료형만 출력이 된다
# Tensor("Const:0", shape=(), dtype=string)

# 그래서 세션을 만들어 실행시켜야 출력이 된다
sess = tf.Session()
print(sess.run(hello)) # b'Hello World'