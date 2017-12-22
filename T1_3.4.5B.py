import tensorflow as tf
import numpy as np
from numpy.random import RandomState
batch_size = 8
w1 = tf.Variable(tf.random_normal([2,4],stddev=1,seed=1))
b1 = tf.Variable(tf.zeros(4))
w2 = tf.Variable(tf.random_normal([4,2],stddev=1,seed=1))
b2 = tf.Variable(tf.zeros(2))

x = tf.placeholder(tf.float32 ,shape=[None,2] ,name='x-input')
y_ = tf.placeholder(tf.float32 ,shape=[None,2] ,name='y-input')

a = tf.nn.sigmoid(tf.matmul(x,w1)+b1)
y = tf.matmul(a,w2)+b2
logits_scaled = tf.nn.softmax(y)

rw = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
cross_entropy = -tf.reduce_sum(y_*tf.log(logits_scaled))+tf.contrib.layers.l2_regularizer(.5)(rw)

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.6,global_step,100,0.98,staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(data_size,2)
Y = [[int(x1+x2<1),int(x1+x2>=1)] for (x1,x2) in X]

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	STEPS = 7000
	for i in range(STEPS):
		start = (i*batch_size)%dataset_size
		end = min(start+batch_size,dataset_size)
		sess.run(train_step,feed_dict = {x : X[start:end],y_ : Y[start:end]})
		if(i%5000==0):
			total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X ,y_:Y})
			print("after %d times , loss is %h"%(i,total_cross_entropy))