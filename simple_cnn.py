# simple cnn wth mnist
# input conv1->relu1->pool1->conv2->relu2->pool2
# 		fc3->relu3->dropout->fc4->softmax->output

from mnist_read import load_mnist

import os, sys
import tensorflow as tf
import numpy as np

mnist = load_mnist()

train_num = 60000
test_num = 10000
batch_size = 100
learning_rate = 0.001
epoch = 4

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
train_mode = tf.placeholder(tf.bool)

conv1 = tf.layers.conv2d(
	inputs = X,
	filters = 32,
	kernel_size = [5,5],
	padding = "same",
	activation = tf.nn.relu)

pool1 = tf.layers.max_pooling2d(
	inputs = conv1,
	pool_size = [2,2], 
	strides = 2)

conv2 = tf.layers.conv2d(
	inputs = pool1,
	filters = 64,
	kernel_size = [5,5],
	padding = "same",
	activation = tf.nn.relu)

pool2 = tf.layers.max_pooling2d(
	inputs = conv2,
	pool_size = [2,2],
	strides = 2)

pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
fc3 = tf.layers.dense(
	inputs = pool2_flat,
	units = 1024,
	activation = tf.nn.relu)

# dropout training -> True : training, False : prediction
dropout3 = tf.layers.dropout(
	inputs = fc3,
	rate = 0.4,
	training = train_mode)

fc4 = tf.layers.dense(
	inputs = dropout3,
	units = 10)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
	logits = fc4,
	labels = Y))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(train_num/batch_size)
## training
for i in range(epoch):
	loss_total = 0
	for j in range(total_batch):
		train_x = mnist['train_img'][0 + (j*batch_size):((j+1)*batch_size)]
		train_y = mnist['train_label'][0 + (j*batch_size):((j+1)*batch_size)]
		_, loss_val = sess.run([train_op, loss],
			feed_dict = {X:train_x, Y:train_y, train_mode:True})
		loss_total += loss_val
		print(" batch %i/%i | loss %f"%(j+1, total_batch, loss_val), end = "\r")

	print("\n epoch : %i  loss : %f"%(i+1, (loss_total/total_batch)))

## test
sum_acc = 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc4,1), tf.argmax(Y,1)),tf.float32))
test_total_batch = int(test_num/batch_size)
for i in range(test_total_batch):
	test_x = mnist['test_img'][0+(i*batch_size):((i+1)*batch_size)]
	test_y = mnist['test_label'][0+(i*batch_size):((i+1)*batch_size)]
	pred = sess.run(accuracy, feed_dict={X:test_x, Y:test_y, train_mode:False})
	sum_acc += pred
	print(" test-num %i/%i acc %f total-acc %f"%(i+1,test_total_batch ,pred, sum_acc), end="\r")
eval_acc = sum_acc/test_total_batch
print(" \n eval-accucary %f"%(eval_acc))
