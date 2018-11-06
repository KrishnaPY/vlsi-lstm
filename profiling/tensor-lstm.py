from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class LSTM_cell(object):
	def __init__(self, inp_dim, out_dim, state_dim):
		self.inp_dim = inp_dim
		self.out_dim = out_dim
		self.state_dim = state_dim
		self.gate_dim = inp_dim+state_dim

		self.x = tf.placeholder(tf.float32,[None, inp_dim])
		self.y = tf.placeholder(tf.float32,[None, out_dim])

		self.Wi = tf.Variable(tf.random_uniform([self.gate_dim,state_dim],
			minval=-np.sqrt(1./self.gate_dim),
			maxval=np.sqrt(1./self.gate_dim)))
		self.Wf = tf.Variable(tf.random_uniform([self.gate_dim,state_dim],
			minval=-np.sqrt(1./self.gate_dim),
			maxval=np.sqrt(1./self.gate_dim)))
		self.Wo = tf.Variable(tf.random_uniform([self.gate_dim,state_dim],
			minval=-np.sqrt(1./self.gate_dim),
			maxval=np.sqrt(1./self.gate_dim)))
		self.Wg = tf.Variable(tf.random_uniform([self.gate_dim,state_dim],
			minval=-np.sqrt(1./self.gate_dim),
			maxval=np.sqrt(1./self.gate_dim)))
		self.Wy = tf.Variable(tf.random_uniform([state_dim,out_dim],
			minval=-np.sqrt(1./state_dim),
			maxval=np.sqrt(1./state_dim)))

		self.Bi = tf.Variable(tf.zeros([state_dim,]))
		self.Bf = tf.Variable(tf.zeros([state_dim,]))
		self.Bo = tf.Variable(tf.zeros([state_dim,]))
		self.Bg = tf.Variable(tf.zeros([state_dim,]))
		self.By = tf.Variable(tf.zeros([out_dim,]))

		self.i = tf.Variable(tf.zeros([state_dim,]))
		self.f = tf.Variable(tf.zeros([state_dim,]))
		self.o = tf.Variable(tf.zeros([state_dim,])) 
		self.g = tf.Variable(tf.zeros([state_dim,])) 

		self.c = tf.Variable(tf.zeros([state_dim,]))
		self.h = tf.Variable(tf.zeros([state_dim,]))

		self.P = tf.Variable(tf.zeros([1,state_dim+inp_dim]))

	def forward_step(self, inputs, output=None):
		self.x = tf.reshape(inputs,(1,-1))
		self.h = tf.reshape(self.h,(1,-1))
		self.P = tf.concat(values=[self.x, self.h],axis=1)
		self.i = tf.nn.sigmoid(tf.matmul(self.P,self.Wi) + self.Bi)
		self.f = tf.nn.sigmoid(tf.matmul(self.P,self.Wf) + self.Bf)
		self.o = tf.nn.sigmoid(tf.matmul(self.P,self.Wo) + self.Bo)
		self.g = tf.nn.tanh(tf.matmul(self.P,self.Wg) + self.Bg)

		self.c = tf.multiply(self.f,self.c) + tf.multiply(self.i,self.g)
		self.h = tf.multiply(self.o,tf.nn.tanh(self.c))


		if(output):
			self.y = tf.matmul(self.h,self.Wy) + self.By
		else:
			self.y = None

		return self.h, self.c, self.y



#Data Setup
data = open('input.txt','r').read()
# data = "Asdfasdfasdfasdfasdfadfa asifnasoidf u"
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_chars = {i:ch for i,ch in enumerate(chars)}
ixData = [char_to_ix[ch] for ch in data]

#Training Parameters
learning_rate = 0.01
num_epochs = 100

#Network Creation
num_hidden = 128
Cell_1 = LSTM_cell(vocab_size, num_hidden, num_hidden)
Cell_2 = LSTM_cell(num_hidden, vocab_size, num_hidden)


character = tf.placeholder(tf.float32,[None, vocab_size])
correct_character = tf.placeholder(tf.int32,[1])

H1, C1, _ = Cell_1.forward_step(character, output='yes')
H2, C2, output = Cell_2.forward_step(H1, output='Yes')

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correct_character,  logits=output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

saver = tf.train.Saver()
builder = tf.profiler.ProfileOptionBuilder
opts = builder(builder.time_and_memory()).order_by('micros').build()

with tf.Session() as sess:
	sess.run(init)
  	saver.restore(sess, "./model1.ckpt")

	for epoch in range(num_epochs):
		error = 0
		for char in range(data_size-1):
			charIN = np.zeros((1,vocab_size))
			charIN[0,ixData[char]] = 1

			charOUT = np.ndarray((1))
			charOUT[0] = ixData[char+1]

			sess.run(train_op, feed_dict={character: charIN, correct_character: charOUT})
			curr_loss = sess.run(loss, feed_dict={character: charIN, correct_character: charOUT})
			error+=curr_loss
		print("Epoch: "+str(epoch)+" Loss: "+ str(error))
	saver.save(sess, "./model2.ckpt")
	print("Optimization Done!")















				

