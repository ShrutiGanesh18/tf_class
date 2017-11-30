import tensorflow as tf
import pickle
import numpy as np

num_epochs = 10000

with tf.name_scope('input_X'):
    X = tf.placeholder(tf.float32,shape=[None,1])
with tf.name_scope('output_Y'):
    Y = tf.placeholder(tf.float32,shape=[None,1])

with open('data/train_ip','r') as f:
    train_ip = pickle.load(f)
    train_ip = np.array(train_ip).reshape(-1,1)
with open('data/train_op','r') as f:
    train_op = pickle.load(f)
    train_op = np.array(train_op).reshape(-1,1)

with open('data/test_ip','r') as f:
    test_ip = pickle.load(f)
    test_ip = np.array(test_ip).reshape(-1,1)
with open('data/test_op','r') as f:
    test_op = pickle.load(f)
    test_op = np.array(test_op).reshape(-1,1)

with tf.name_scope('grad_vars'):
    W = tf.Variable(tf.random_uniform(shape=[1,1],dtype=tf.float32))
    b = tf.Variable([30.0])
y = tf.matmul(X,W)+ b

# mean_squared_error loss
cost = tf.reduce_mean(tf.squared_difference(Y, y))
learn = tf.train.GradientDescentOptimizer(0.0000004).minimize(cost)

tf.summary.scalar('cost_summary',cost)
merged = tf.summary.merge_all()


with tf.Session() as session:
    train_writer = tf.summary.FileWriter('/train', session.graph)
    session.run(tf.global_variables_initializer())

    for i in range(num_epochs):
        s,_ = session.run([merged,learn],feed_dict={X:train_ip,Y:train_op})

        if i%100 == 0:
            acc = session.run(cost,feed_dict={X:test_ip,Y:test_op})
            print 'epoch: ' + str(i) + '\tloss: ' + str(acc)


        train_writer.add_summary(s,i)
