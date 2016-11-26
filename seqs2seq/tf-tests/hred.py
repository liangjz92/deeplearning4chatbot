#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import tensorflow as tf
import numpy as np
class HRED:
	def __init__(self):
		self.train = True
		self.memory_size = 100
		self.vocab_size = 1000
		self.embedding_size = 300
		self.max_dialog_size = 6	#最长50次交互
		self.max_sentence_size = 10	#每句话长度为100个单词
		self.num_samples = 50	#带采样的softmax
		self.learning_rate = tf.Variable(float(0.5),trainable =False,dtype= tf.float32)
		self.learning_rate_decay_factor = 0.99
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)
		self.max_gradient_norm =5.0
	
	def build_model(self):
		#self.embedding_weight = tf.get_variables('embedding_weight',[self.vocab_size,self.embedding_size],dtype=tf.float32)
		#词向量权重
		self.encoders = []	#完整对话的输入参数
		self.decoders = []	#用于解码的输入
		self.weights = []	#指示解码的输入中哪些是补全的
		self.targets = []	#decoder应该输出的正确答案
		

		for i in range(self.max_dialog_size):	#这么多句话
			encoder = []
			for j in range (self.max_sentence_size):	#每句话这么多单词
				encoder.append(tf.placeholder(tf.int32,shape=[None],name="encoder_{0}_{1}".format(i,j)))
			self.encoders.append(encoder)

			if i%2==0:	#有一半的对话是需要解码的
				index=int(i/2)
				decoder = []
				weight = []
				for j in range(self.max_sentence_size):
					decoder.append(tf.placeholder(tf.int32,shape=[None],name = "decoder_{0}_{1}".format(index,j)))
					weight.append(tf.placeholder(tf.float32,shape=[None],name="weight_{0}_{1}".format(index,j)))
				self.decoders.append(decoder)
				self.weights.append(weight)
				target= [self.decoders[index][j] for j in xrange(self.max_sentence_size)]	#targets是decoder input向后顺延一位的结果
				self.targets.append(target)
		
		###################################################
		self.encoder_outputs=[]	#各个编码器在最后一时刻的输出结果
		self.context_outputs=None	#上下文编码器的数据结果
		self.decoder_outputs=[]	#各个解码器的解码结果，用以计算损失函数

		with tf.variable_scope("encoder") as scope:
			for i in range(self.max_dialog_size):
				cell = tf.nn.rnn_cell.GRUCell(self.memory_size)
				cell = tf.nn.rnn_cell.EmbeddingWrapper(cell,self.vocab_size,self.embedding_size)
				(outputs,_) = tf.nn.rnn(cell,self.encoders[i],dtype=tf.float32)	#创建一个rnn来执行
				self.encoder_outputs.append(outputs[-1])	#提取最后一时刻的输出进行保存

				if i==0:	#第一次执行结束之后，开启重用变量的功能
					tf.get_variable_scope().reuse_variables()

		with tf.variable_scope("context") as scope:
			cell = tf.nn.rnn_cell.GRUCell(self.memory_size)	#用来编码上下文的rnn cell,没有任何特别之处
			(self.context_outputs,_) = tf.nn.rnn(cell,self.encoder_outputs,dtype=tf.float32)	#直接在encoder的结果上进行编码

		with tf.variable_scope("decoder") as scope:
			for i in range(int(self.max_dialog_size/2)):
				cell = tf.nn.rnn_cell.GRUCell(self.memory_size)
				cell = tf.nn.rnn_cell.EmbeddingWrapper(cell,self.vocab_size,self.embedding_size)
				(decoder_output,_)=tf.nn.rnn(cell,self.decoders[i],initial_state = self.context_outputs[i],dtype=tf.float32)
				self.decoder_outputs.append(decoder_output)
				if i==0:
					tf.get_variable_scope().reuse_variables()

		###################
		output_projection = None
		softmax_loss_function = None
		if self.num_samples >0 and self.num_samples <self.vocab_size:
			w_t =tf.get_variable('proj_w',[self.vocab_size,self.memory_size],dtype=tf.float32)
			w =tf.transpose(w_t)
			b =tf.get_variable('proj_b',[self.vocab_size],dtype=tf.float32)
			output_projection = (w,b)
			
			def sampled_loss(inputs,labels):
				labels = tf.reshape(labels,[-1,1])
				return tf.nn.sampled_soft_max_loss(w_t,b,inputs,labels,self.num_samples,self.vocab_size)
				
		softmax_loss_function =  sampled_loss
		losses = []
		for i in range(int(self.max_dialog_size/2)):
			loss = tf.nn.seq2seq.sequence_loss(
					self.decoder_outputs[i],	#rnn解码器输出的结果，[ batch*symbol , ...]
					self.targets[i],	#预期的单词		[batch, ....]
					self.weights[i],	#计算loss时的权重 [batch, ...]
					softmax_loss_function
					)
			losses.append(loss)
		loss_sum = tf.reduce_sum(tf.pack(losses))	#final loss
		
		
		
		
		self.gradient_norms = []
		self.updates=[]
		opt = tf.train.GradientDescentOptimizer(self.learning_rate)
		params = tf.trainable_variables()
		gradients = tf.gradients(loss_sum,params)
		clipped_gradients, norm = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
		self.gradient_norms.append(norm)
		self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
		self.saver = tf.train.Saver(tf.all_variables())
			
		
			
				
				
										
if __name__ == '__main__':
	model = HRED()
	init = tf.initialize_all_variables()	
	with tf.Session() as sess:
		sess.run(init)
		model.build_model()
#		sess.run(data,feed_dict={data:np.array([[[1.0,2.0]]])})
#		print(sess.run(output,feed_dict={data:np.array([[[1.0,2.0]]])}))
