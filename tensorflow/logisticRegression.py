import tensorflow as tf
import numpy as np
tf.set_random_seed(888)

xy = np.loadtxt('../data/04train.txt', dtype='float32')

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W= tf.Variable(tf.random_uniform([3,1],-1., 1.))

h = tf.matmul(X, W)
hypothesis = tf.div(1., 1. + tf.exp(-h))

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={ X:x_data, Y:y_data })
    if step % 20 == 0 :
        print(step, sess.run(cost, feed_dict={ X:x_data, Y:y_data }), sess.run(W))


a = sess.run(hypothesis, feed_dict={X:[[1, 2, 2]]})
print("=====prediction====", a)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast( tf.equal(predicted, Y), dtype=tf.float32))
acc = sess.run(accuracy, feed_dict = {X : x_data, Y : y_data})
print(acc)