# tf06_2.py 의 lr 을 수정해서
# 에폭 2000 전에 수렴하게 만들어라

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

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= 0.5).minimize(loss)
# 0.01 - 3820에서 동결 loss = 1.2065964e-10
# 0.05 - 900에서 동결 loss = 4.49063e-12
# 0.1 - 500에서 동결 loss = 3.7895612e-13

# 0.1741 - 100 2.3349254e-05 [1.9999194] [1.0046744]
# 0.1742 - 100 2.8959803e-05 [2.0001352] [1.0047578]
# 0.17415 - 100 2.5989744e-05 [2.000024] [1.0047148]
# 0.174125 - 100 2.4625593e-05 [1.9999708] [1.0046942]
# 0.1741375 100 2.5290627e-05 [1.999997] [1.0047042]
# 0.17413875 100 2.536112e-05 [1.9999998] [1.0047054]
# 0.17413885 100 2.5362526e-05 [1.9999999] [1.0047052]
# 0.17413885119 100 2.5362526e-05 [1.9999999] [1.0047052]
         
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        # 최종 결과 세션 돌릴때 피드딕트 돌리면 된다!
        _, cost_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={x_train:[1,2,3], y_train : [3,5,7]})
        if step %1 == 0:
            print(step, cost_val, W_val, b_val)
    
    # predict
    # print('[4] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[4]}))
    # print('[5, 6] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[5,6]}))
    # print('[6, 7, 8] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[6,7,8]}))

# [실습]
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]
