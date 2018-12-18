import tensorflow as tf
import numpy as np
tf.set_random_seed(888)

xy = np.loadtxt('../data/05train.txt', dtype='float32')

x_data = xy[:, 0:3]
y_data = xy[:, 3:]

print(x_data.shape, y_data.shape)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W= tf.Variable(tf.zeros([3,3]))
hypothesis = tf.nn.softmax(tf.matmul(X,W))
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for step in range(5001):
        sess.run(train, feed_dict={ X:x_data, Y:y_data })

        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={ X:x_data, Y:y_data }), sess.run(W))

    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print("=======prediction=======", a, sess.run(tf.argmax(a, 1)))


    c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
    print("=======prediction=======", a, sess.run(tf.argmax(c, 1)))

    correct_predicted = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predicted, dtype=tf.float32))
    acc = sess.run(accuracy, feed_dict={X: x_data, Y: y_data})
    print(acc)