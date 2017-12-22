import tensorflow as tf

import numpy as np
from numpy.random import RandomState

batch_size=8

w1 = tf.Variable(tf.random_normal([2,4],stddev=1,seed=1))

b1 = tf.Variable(tf.zeros([4]))
w2 = tf.Variable(tf.random_normal([4,2],stddev=1,seed=1))


b2 = tf.Variable(tf.zeros([2]))
#b2 = np.array([0.71348143]);


w1__ = [[-1.9618274,2.58235407,1.68203783],[-3.46817183,1.06982327,2.11789012]]


w2__ = [[-1.82471502]
,[ 2.68546653]
,[ 1.41819513]]

#w1__ = [[-1.37271583,2.0234375,0.92637938],[-2.92271328,0.55797005,1.38236868]]
#w2__ = [[-1.29980099],[ 2.06771374],[ 0.7132026 ]]
#b2__ = [0.71348143]
w1__ = [[-0.81131822,1.48459876,0.06532937], [-2.4427042,0.0992484,0.59122431]]
w2__ = [[-0.81131822], [ 1.48459876], [ 0.06532937]]
w1__ = [[ 3.52762985 , 3.06831455 , 3.9016757 ], [ 2.25749707 , 1.83648407 , 4.59318638]]
w2__ = [[-6.51849651], [-3.35110521], [-4.84551811]]

w1__ = [[-0.10416051 , 0.77436566 , 1.18454385], [-2.0415225 , -0.33159316 , 1.18694019]]
w2__ = [[-0.35187641 , 1.02515686], [-0.67082113, -1.70656192], [-0.72347498 , 1.41394937]]

w1__ = [[ 27.40246582 , -5.03092194 , -0.24702913, -22.48351669], [20.5919857 ,  -5.51003027 , -0.22443163 ,-22.31588554]]
w2__ = [[ -9.77084935e-01 , -1.40028250e+00], [ -9.77084935e-01 , -1.40028250e+00], [2.56927386e-02  , 6.64764285e-01], [ 3.92339630e+01 , -4.07640724e+01]]

w1__ = [[ 27.37519646 , -5.15104437 , -0.25831696 ,-22.49417305], [ 20.58813667 , -5.54119921 , -0.3013764 , -22.29509354]]
w2__ = [[-19.30745316 , 19.98015976] ,[ -1.09360611 , -1.28377104] ,[  0.05966205  , 0.63078523] ,[ 38.92435074 ,-40.45447159]]
b1__ = [-22.21904564 , -3.92961574 , -6.55378008 , 22.08110619]
b2__ = [-5.6088748 , 5.6087923]
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')

y_ = tf.placeholder(tf.float32,shape=(None,2),name='y-input')




#a = tf.matmul(x,w1)+b1
#a = tf.nn.sigmoid(tf.matmul(x,w1))
a = tf.nn.sigmoid(tf.matmul(x,w1)+b1)

y = tf.matmul(a,w2)+b2
#y = tf.nn.sigmoid(tf.matmul(a,w2))
#y = tf.matmul(a,w2)+b2
logits_scaled = tf.nn.softmax(y)

#cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
ww = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
cross_entropy = -tf.reduce_sum(y_ * tf.log(logits_scaled))+tf.contrib.layers.l2_regularizer(.5)(ww)

#cross_entropy = -tf.reduce_sum(y_ * tf.log(logits_scaled),1)


#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.6,global_step,100,0.98,staircase=True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy , global_step = global_step)


rdm = RandomState(1)

dataset_size = 128

X = rdm.rand(dataset_size ,2)

X = X.astype(np.float32)

Y = [[int(x1+x2<1),int(x1+x2>=1)] for(x1,x2) in X]


a__ = tf.nn.sigmoid(tf.matmul(X.astype(np.float32),w1__)+b1__)
y__ = tf.matmul(a__,w2__)+b2
#y__ = tf.matmul(a__,w2__)+b2__

logits_scaled__ = tf.nn.softmax(y__)
clip_by_value__ = tf.clip_by_value(y__,1e-10,1.0)
#clip_by_value__ = y__
log__=tf.log(clip_by_value__)
mul__=Y*log__


with tf.Session() as sess:

	init_op = tf.global_variables_initializer()#initialize_all_variables()
 
	sess.run(init_op)

	print(sess.run(w1))

	print(sess.run(w2))

	print('X',X[0:16])
	print('Y',Y[0:16])
	y1__ = sess.run(y__)
	y___ = []
	for i,v in enumerate(y1__):
		y___.append(v[0])
	#print('y___',y___)
	#cross_entropy__ = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(y___,1e-10,1.0)))
	print('y__',sess.run(y__)[0:16],sess.run(tf.nn.softmax(y__)))
	#print('clip_by_value__',sess.run(clip_by_value__)[0:16])
	#print('log__',sess.run(log__)[0:16])
	#print('mul__',sess.run(mul__)[0:16])
	#print('cross_entropy__',sess.run(cross_entropy__))
	'''

	w1 = [[-0.81131822  1.48459876  0.06532937]
 [-2.4427042   0.0992484   0.59122431]]


	w2 = [[-0.81131822]
 [ 1.48459876]
 [ 0.06532937]]


	'''

	STEPS =70000

	for i in range(STEPS):

		start = (i * batch_size)%dataset_size

		end = min(start+batch_size,dataset_size)

		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
	
		if i%5000 ==0:
	
			total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})

			#print("After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))


			#print("After %d training step(s),cross entropy on all data is"%(i),sess.run(cross_entropy))
			print(sess.run(w1),sess.run(b1),sess.run(w2),sess.run(b2))
			print("After %d training step(s),cross entropy on all data is"%(i),total_cross_entropy)
	print(sess.run(w1),sess.run(b1))

	print(sess.run(w2),sess.run(b2))

	#print(sess.run(total_cross_entropy ))

	print('total_cross_entropy',total_cross_entropy )
	'''

	w1 = [[-1.9618274,2.58235407,1.68203783],[-3.46817183,1.06982327,2.11789012]]


	w2 = [[-1.82471502]
,[ 2.68546653]
,[ 1.41819513]]

	'''
