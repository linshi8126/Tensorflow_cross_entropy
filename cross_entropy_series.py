#https://zhuanlan.zhihu.com/p/27842203
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
labels = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=np.float32)
logits = np.array([[1, 2, 7],
                   [3, 5, 2],
                   [6, 1, 3],
                   [8, 2, 0],
                   [3, 6, 1]], dtype=np.float32)

num_classes = labels.shape[1]
predicts = tf.nn.softmax(logits=logits, dim=-1)
classes = tf.argmax(labels, axis=1)

print(num_classes)
print('softmax predicts:',sess.run(predicts))
print('classes /tf.argmax:',sess.run(classes))

labels = tf.clip_by_value(labels, 1e-10, 1.0)
predicts = tf.clip_by_value(predicts, 1e-10, 1.0)
cross_entropy1 = tf.reduce_sum(labels * tf.log(labels/predicts), axis=1)
cross_entropy1A = -tf.reduce_sum(labels * tf.log(predicts), axis=1)
cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
cross_entropy3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=classes)
z = 0.8
x = 1.3
cross_entropy4 = tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=x)
# tf.nn.sigmoid_cross_entropy_with_logits的具体实现:
cross_entropy5 = - z * tf.log(tf.nn.sigmoid(x)) - (1-z) * tf.log(1-tf.nn.sigmoid(x))
print('labels/predicts:',sess.run(labels/predicts))
print('cross_entropy1:',sess.run(cross_entropy1))
print('cross_entropy1A:',sess.run(cross_entropy1A))
print('cross_entropy2:',sess.run(cross_entropy2))
print('cross_entropy3:',sess.run(cross_entropy3))
print('cross_entropy4:',sess.run(cross_entropy4))
print('cross_entropy5:',sess.run(cross_entropy5))