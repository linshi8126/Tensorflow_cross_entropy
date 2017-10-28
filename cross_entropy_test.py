'''
一. 交叉熵，用于分类问题：
4分类问题，数据的批大小为2，所以此处标签可以用 2*4的矩阵表示，特别注意：输出层为softmax层。
y_：数据的正确标签，用one hot码表示
y1：第一批输入softmax层的激活值
y2：第二批输入softmax层的激活值
output1：第一批的最终输出结果
output2：~
cross_entropy1：使用y1的交叉熵
cross_entropy2：~
这里说明一下为什么使用softmax层作为输出层：
1. 使用softmax即假设每种分类之间是相互对立的，因此所有的输出值相加为1（即输出的所有概率之和为1），如果没有相互对立的假设，则不能使用softmax
（例如：当对一个人按肤色进行分类预测时，由于一个人只有可能是黄、白、黑三种之一，因此 p(白) + p(黄) + p(黑)=1，此时可以使用softmax作为输出层）；
2. softmax可以令每一个元素属于（0,1）的范围内，从而避免log0发生。
'''

import tensorflow as tf  
y_ = tf.constant([1,0,0,0,0,0,0,1],dtype=tf.float32,shape=[2,4])  
y1 = tf.constant([1,0,0,0,0,0,0,1],dtype=tf.float32,shape=[2,4])  
y2 = tf.constant([5,0,0,0,0,0,0,10],dtype=tf.float32,shape=[2,4])  

z_ = tf.constant([1,2,3],dtype=tf.float32,shape=[3,1])  
z1 = tf.constant([1,2,3],dtype=tf.float32,shape=[3,1])  
z2 = tf.constant([2,3,4],dtype=tf.float32,shape=[3,1])  

output1 = tf.nn.softmax(y1)  
output2 = tf.nn.softmax(y2)  
cross_entropy1 = -tf.reduce_mean(y_*tf.log(output1))  
cross_entropy2 = -tf.reduce_mean(y_*tf.log(output2))  

MSE1 = tf.reduce_mean(tf.square(z1-z_))  
MSE2 = tf.reduce_mean(tf.square(z2-z_))  

sess = tf.InteractiveSession()  

print('output 1:',output1.eval())  
print('output 2:',output2.eval())  
print('cross entropy 1:',cross_entropy1.eval())  
print('cross entropy 2:',cross_entropy2.eval())  

print('MSE 1:',MSE1.eval())  
print('MSE 2:',MSE2.eval())  