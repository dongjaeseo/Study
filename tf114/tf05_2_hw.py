import tensorflow as tf
tf.set_random_seed(66)

x_train = [[1.,2.],[3.,4.]]
y_train = [[1.,2.],[3.,4.]]

W = tf.Variable(tf.random_normal([2,2]), name = 'weight')
b = tf.Variable(tf.random_normal([1,2]), name = 'bias')

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print('\ninitial weight : ', sess.run(W), '\ninitial bias : ', sess.run(b))

hypothesis = tf.matmul(x_train, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # 로스함수 / mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(3):
    sess.run(train)
    if step %1 == 0:
        print('\nstep : ', step, '\ncost : ', sess.run(cost), '\nweight : ', sess.run(W), '\nbias : ', sess.run(b))
