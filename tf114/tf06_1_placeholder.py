# [실습]
# placeholder 사용

import tensorflow as tf
tf.set_random_seed(66)

#### 여기가 바뀐거에요!!!!
# 기존 1 2 3 3 5 7 대신 쉐이프 none 을 가진 placeholder 사용
# 쉐이프 none 은 자유로운 쉐이프를 받는다
x_train = tf.placeholder(tf.float32, shape = [None])
y_train = tf.placeholder(tf.float32, shape = [None])

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(W), sess.run(b))

loss = tf.reduce_mean(tf.square(x_train * W + b - y_train)) # 로스함수 / mse

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3):
        # 최종 결과 세션 돌릴때 피드딕트 돌리면 된다!
        sess.run(train, feed_dict= {x_train:[1,2,3], y_train : [3,5,7]})
        if step %1 == 0:
            # 마찬가지로 x, y (플레이스홀더) 데이터를 사용하는 세션에 피드딕트 넣어준다!
            print(step, sess.run(loss, feed_dict= {x_train:[1,2,3], y_train : [3,5,7]}), sess.run(W), sess.run(b))