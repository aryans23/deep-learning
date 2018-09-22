import tensorflow as tf
mnist = tf.keras.datasets.mnist
from keras.utils import np_utils
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')

# more layers
# W11 = tf.Variable(tf.random_normal([600, 500], stddev=0.03), name='W11')
# b11 = tf.Variable(tf.random_normal([500]), name='b11')
# W12 = tf.Variable(tf.random_normal([500, 300], stddev=0.03), name='W12')
# b12 = tf.Variable(tf.random_normal([300]), name='b12')
# and the weights connecting the hidden layer to the output layer

W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# calculate the output of the hidden layer
h1 = tf.add(tf.matmul(x,W1), b1)
a1 = tf.nn.relu(h1)
# h11 = tf.add(tf.matmul(a1,W11), b11)
# a11 = tf.nn.relu(h11)
# h12 = tf.add(tf.matmul(a11,W12), b12)
# a12 = tf.nn.relu(h12)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
# y_ = tf.nn.softmax(tf.add(tf.matmul(a12, W2), b2))
y_ = tf.nn.softmax(tf.add(tf.matmul(a1, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], 
                         feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

