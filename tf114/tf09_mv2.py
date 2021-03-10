import tensorflow as tf
tf.set_random_seed(66)

# x_data = [[73, 51, 65],
#           [92, 98, 11],
#           [89, 31, 33],
#           [99, 33, 100],
#           [17, 66, 79]]

x_data = [[73, 80, 75],
          [93, 88, 93],
          [85, 91, 90],
          [96, 98, 100],
          [73, 66, 70]]

y_data = [[152], [185], [180], [196], [142]]

x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

# [실습] 맹그러봐
loss = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(learning_rate= 0.000025).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(10001):
        _, cur_loss, cur_hypothesis, cur_w, cur_b = sess.run([train, loss, hypothesis, w, b], feed_dict= {x:x_data, y:y_data})
        if epoch%20 == 0:
            print(f'Epoch : {epoch} >>> loss : {cur_loss}\nhypo : {cur_hypothesis}')