import tensorflow as tf
logits_list = [tf.Variable([[1.0, 2.0, 3.0]]), tf.Variable([[1.0, 5.0, 6.0]])]
labels1 = tf.Variable([[7.0, 8.0, 9.0]])
labels2 = tf.cast(tf.Variable([1]), tf.int64)
loss_list1 = [tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels1)for logits in logits_list]
loss_list2 = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels2)for logits in logits_list]

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(loss_list1))
	print(sess.run(loss_list2))

import math
part11=-math.log(math.exp(1)/(math.exp(1)+math.exp(2)+math.exp(3)))
part12=-math.log(math.exp(2)/(math.exp(1)+math.exp(2)+math.exp(3)))
part13=-math.log(math.exp(3)/(math.exp(1)+math.exp(2)+math.exp(3)))
part1=7*part11+8*part12+9*part13
print(part1)

part21=-math.log(math.exp(1)/(math.exp(1)+math.exp(5)+math.exp(6)))
part22=-math.log(math.exp(5)/(math.exp(1)+math.exp(5)+math.exp(6)))
part23=-math.log(math.exp(6)/(math.exp(1)+math.exp(5)+math.exp(6)))
part2=7*part21+8*part22+9*part23
print(part2)