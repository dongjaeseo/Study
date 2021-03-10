import tensorflow as tf

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b

# [실습]
#1. sess.run()
#2. InteractiveSession
#3. eval(session = sess)
# hypothesis 를 출력하는 코드를 만드시오

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('sess.run() : ', sess.run(hypothesis))

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('sess.run() : ', hypothesis.eval(session = sess))

# tf.compat.v1.Session() 은 .as_default() 가 기본으로 되어있는데
# tf.compat.v1.InteractiveSession() 은 .as_default()를 붙여줘야한다
with tf.compat.v1.InteractiveSession().as_default() as sess:
    sess.run(tf.global_variables_initializer())
    print('eval() : ', hypothesis.eval())
    # 인터랙티브는 꺼줘야한다!!
    sess.close()