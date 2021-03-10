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

hypothesis = x_train * W + b
loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # 로스함수 / mse

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(loss)
         
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        # 최종 결과 세션 돌릴때 피드딕트 돌리면 된다!
        _, cost_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={x_train:[1,2,3], y_train : [3,5,7]})
        if step %20 == 0:
            print(step, cost_val, W_val, b_val)
    
    # predict
    print('[4] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[4]}))
    print('[5, 6] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[5,6]}))
    print('[6, 7, 8] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[6,7,8]}))

# [실습]
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]
