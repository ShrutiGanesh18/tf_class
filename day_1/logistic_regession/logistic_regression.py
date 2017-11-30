import tensorflow as tf
import numpy as np
import pickle
num_epochs = 1000

X = tf.placeholder(tf.float32,shape=[None,2])
Z = tf.placeholder(tf.float32,shape=[None,2])

with open('data/train_ip','r') as f:
    train_ip = pickle.load(f)
    train_ip = np.array(train_ip).reshape(-1,2)
with open('data/train_op','r') as f:
    train_op = pickle.load(f)
    train_op = np.eye(2)[np.array(train_op)]

with open('data/test_ip','r') as f:
    test_ip = pickle.load(f)
    test_ip = np.array(test_ip).reshape(-1,2)
with open('data/test_op','r') as f:
    test_op = pickle.load(f)
    test_op = np.eye(2)[np.array(test_op)]

W = tf.Variable(tf.random_uniform(shape=[2,2],dtype=tf.float32))
b = tf.Variable([20.0,0.0])

z = tf.matmul(X, W) + b#tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Z,logits=z))
#cost = tf.reduce_mean(-tf.reduce_sum(Z*tf.log(z), reduction_indices=1))
learn = tf.train.GradientDescentOptimizer(0.000009).minimize(cost)

correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(Z, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(num_epochs):
        session.run(learn,feed_dict={X:train_ip,Z:train_op})

        if i%50 == 0:
            acc = session.run(accuracy,feed_dict={X:test_ip,Z:test_op})
            print 'epoch: ' + str(i) + '\tacc: ' + str(acc)
