import tensorflow as tf

sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype = tf.float32, name = 'test')

# 변수는 sess.run 을 하기 전에 무조건 초기화를 거쳐야 한다
# 소스코드에 있는 모든 변수를 초기화해준다
# 값이 바뀌는건 X >> 텐서플로에 맞게 변경시켜준다 
init = tf.compat.v1.global_variables_initializer()

sess.run(init)

print(sess.run(x))