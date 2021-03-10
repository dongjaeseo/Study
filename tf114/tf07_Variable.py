import tensorflow as tf
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.random.normal([1]), name ='weight')
print(W)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(W)
print(aaa)
sess.close()

# sess.run = eval()
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = W.eval()
print(bbb)
sess.close()

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = W.eval(session = sess)
print(ccc)
sess.close()