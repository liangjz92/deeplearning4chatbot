#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import data_utils
class HREDModel:
#层次化的RNN模型，内部单元默认使用GRU
#输入包括多句话，各句话均使用相同的RNN进行编码，最终输出一个定长向量，作为句子的表达
#一个更高层的LSTM不断读入上面的向量，并生成context的状态
#每当用户的对话说完之后，一个decoder根据当前的context向量产生一句话
#吐槽：目前的损失函数的估量太粗糙
	def __init__(self):
		self.vocab_size = 10000		#字典长度
		self.memory_size = 300		#RNN单元长度
		self.embedding_size = 300	#词向量维度
		self.dialog_length = 30		#最长支持的会话（两个人交替说话）
		self.user_sentence_length = 50		#用户的描述，最长长度
		self.waiter_sentence_length = 200	#客服的话，最长长度
		self.batch_size = 5					#还是要想办法实现batch?
		self.learning_rate = tf.Variable(float(0.5), trainable=False)	#学习率
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.995)	#学习率递减
		self.global_step = tf.Variable(0, trainable=False)	#记录训练了多少次
		output_projection = None		#decoder在解析过程中需要将维度变为字典长度，以进行softmax
		softmax_loss_function = None	#softmax损失函数
		num_samples = 512				#采样softmax	
		#######################
		if num_samples > 0 and num_samples<self.target_vocab_size:	
			#当sampled_softmax的最高采样数小于字典长度时，需要提供一个映射的权值矩阵
			#仅使用在解码器中使用
			w = tf.get_variable('proj_w',[self.memory_size,self.vocab_size])
			w_t = tf.transpose(w)
			b = tf.get_variable('proj_b',[self.vocab_size])
			output_projection = (w,b)
			def sample_loss(inputs,labels):
				labels = tf.reshape(labels,[-1,1])	#将labels进行展开
				local_w_t = tf.cast(w_t,tf.float32)
				local_b = tf.cast(b,tf.float32)
				local_inputs = tf.cast(inputs,tf.float32)
				return tf.nn.sampled_softmax_loss(local_w_t,local_b,local_inputs,labels,num_samples,self.target_vocab_size)
			softmax_loss_function = sample_loss
		########################
		#定义使用的各个单元
		encoder_cells = []	#用于编码的encoder_cell 相互之间参数共享
		decoder_cells = []	#用于解码的decoder_cell 相互之间参数共享
		context_cell =tf.nn.rnn_cell.GRUCell(self.memory_size)	#用于编码上下文的RNNcell，独此一份无需共享
		with tf.variables_scope('hred_encoder'):
			for i in range(self.dialog_length):
				context_cell =tf.nn.rnn_cell.GRUCell(self.memory_size)	#用于编码上下文的RNNcell，独此一份无需共享
		#300维的GRUCell
		######################
		self.encoder_inputs = []
		self.decoder_inputs = []
		self.target_weights = []
		for i in xrange(self.encoder_length):
			self.encoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name='encoder{0}'.format(i)))
		for i in xrange(self.decoder_length):
			self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
			self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
		targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]
		######################
		self.outputs, self.states = tf.nn.seq2seq.embedding_rnn_seq2seq(
			self.encoder_inputs,
			self.decoder_inputs, 
			single_cell,
			num_encoder_symbols=self.source_vocab_size,
			num_decoder_symbols=self.target_vocab_size,
			embedding_size=size,
			output_projection=output_projection,
			feed_previous=False
		)
		self.losses = tf.nn.seq2seq.sequence_loss(
			self.outputs[:-1],
			targets,
			self.target_weights[:-1],
			softmax_loss_function=softmax_loss_function
		)
if __name__ == '__main__':
	test = TestModel()
	print("开始执行")
