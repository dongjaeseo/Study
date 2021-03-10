import tensorflow as tf
tf.set_random_seed(66)

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(W), sess.run(b))

hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # 로스함수 / mse

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(loss)

'''
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
for step in range(3):
    sess.run(train)
    if step %1 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))

sess.close()
'''
# 세션은 끝나고 수동으로 닫아주는게 좋다 근데 번거로우니 with 문을 써보자
# 위와 동일 이렇게 하면 위드문이 끝날때 세션이 알아서 종료된다
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3):
        sess.run(train)
        if step %1 == 0:
            print(step, sess.run(loss), sess.run(W), sess.run(b))