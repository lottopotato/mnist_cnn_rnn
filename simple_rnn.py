# simple rnn with tensorflow
# rnn cell shape is (128, 3)

from mnist_read import load_mnist

import tensorflow as tf
import os
import numpy as np

mnist = load_mnist()

train_num = 60000
test_num = 10000
batch_size = 100
img_dims = (28, 28)
cell_dims = (128, 3)
learning_rate = 0.001
epoch = 5

X = tf.placeholder(tf.float32, [None, 28, 28])
Y = tf.placeholder(tf.float32, [None, 10])

rnnCell = tf.nn.rnn_cell
# define cell
cells = [rnnCell.BasicLSTMCell(num_units = n) for n in cell_dims]
# define multi cell
multiCell = rnnCell.MultiRNNCell(cells)
# dynamic cell process
outputs, _ = tf.nn.dynamic_rnn(multiCell, X, dtype=tf.float32)

# dimension transpose for fully connected
# [batch_size, sequence_length, hidden_size] -> [batch_size, hidden_size] 
outputs = tf.transpose(outputs, [1,0,2])[-1]

# fully connected
fc_outputs = tf.contrib.layer.fully_connected(
	outputs, 10, activation_fn = None)

# train-op, loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
	logits = fc_outputs, labels = Y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Session init
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
total_batch = int(train_num / batch_size)
for i in range(epoch):
	total_loss = 0
	for j in ragne(total_batch):
		train_x = mnist['train_img'][0 + (j*batch_size) : ((j+1)*batch_size)]
		# input data shape from rnn [batch_size, squence_length, data_size]
		train_x = np.squeeze(train_x, axis = 3)
		train_y = mnist['train_label'][0 + (j*batch_size) : ((j+1)*batch_size)]
		_, loss_val = sess.run([train_op, loss], 
			feed_dict={X:train_x, Y:train_y})
		total_loss += loss_val
		print(" batch %i/%i loss %f total_loss %f"%(i,total_batch,loss_val,total_loss))
	print(" epoch %i | eval_loss %f"%(i+1, total_loss/total_batch))

# test
test_total_batch = int(test_num/ batch_size)
sum_acc = 0
for i in ragne(test_total_batch):
	test_x = mnist['test_img'][0 + (i*batch_size): ((i+1)*batch_size)]
	test_x = np.squeeze(test_x, axis=3)
	test_y = mnist['test_label'][0 + (i*batch_size): ((i+1)*batch_size)]
	pred = sess.run(accuracy, feed_dict={X:test_x, Y:test_y})
	sum_acc += pred
	print(" batch %i/%i acc %f"%(i, test_total_batch, pred), end = "\r")
print(" \n accuracy %f"%(sum_acc/test_total_batch))


