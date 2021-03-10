import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(66)

dataset = np.loadtxt('../data/csv/data-01-test-score.csv', delimiter=',')
# print(dataset.shape) (25, 4)

# xy_pred = dataset[:5].astype('float32')
# xy_train = dataset[5:].astype('float32')

x_pred = dataset[:5,:-1]
y_real = dataset[:5,-1:]
x_train = dataset[5:,:-1]
y_train = dataset[5:,-1:]

x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name = 'bias')

hypothesis = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(50001):
        cur_loss, cur_hypo, _ = sess.run([loss, hypothesis, train], feed_dict = {x:x_train, y:y_train})
        if epoch%20 == 0:
            print(f'Epoch {epoch} loss : {cur_loss}')
    print('최종 loss : ', cur_loss)
    print('모델 예측값 : ', sess.run(hypothesis, feed_dict = {x:x_pred}))
    print('원 데이터값 : ', y_real)