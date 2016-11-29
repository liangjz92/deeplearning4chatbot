#coding=utf-8
import tensorflow as tf
import numpy as np

class Ranker:
	def __init__(self,
			vocab_size = 20001,
			embedding_size = 300,
			batch_size =20,
			max_dialogue_size = 25,
			max_sentece_size = 36,
			margin = 0.05,
			learning_rate =1.0,
			learning_rate_decay_factor = 0.9,
			use_lstm = False,
			train_mode = True
			):
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size 
		self.batch_size = batch_size
		self.max_dialogue_size = max_dialogue_size
		self.max_sentence_size = max_sentence_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
		self.use_lstm = use_lstm 
		self.train_mode =train_model 
		self.global_step = tf.Variable(0, trainable=False)

	def build_model(self):
		self.history_index = []
		self.true_index = []
		self.false_index = []
		self.history_emd = []
		self.true_emd = []
		self.false_emd = []
		self.embedding_weight = tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0),name = 'embedding_size')
		for i in range (self.max_dialogue_size):
			#创建历史对话记录的部分
			index = []
			emd = []
			for j in range(self.max_sentece_size):
				index.append(tf.placeholder(tf.int32,shape=[None],name="history_index_{0}_{1}".format(i,j)))
				emd.append(tf.nn.embedding_lookup(self.embedding_weight,index[-1]))
			self.history_index.append(index)
			self.history_emd.append(emd)

		for i in range (int(self.max_dialogue_size/2)):
			#创建待匹配的部分的输入
			tindex = []
			temd = []
			findex= []
			femd = []
			for j in range(self.max_sentece_size):
				tindex.append(tf.placeholder(tf.int32,shape=[None],name="true_index_{0}_{1}".format(i,j)))
				findex.append(tf.placeholder(tf.int32,shape=[None],name="false_index_{0}_{1}".format(i,j)))
				temd.append(tf.nn.embedding_lookup(self.embedding_weight,tindex[-1]))
				femd.append(tf.nn.embedding_lookup(self.embedding_weight,findex[-1]))

			self.true_index.append(tindex)
			self.true_emd.append(temd)
			self.false_index.append(findex)
			self.false_emd.append(femd)
		########################################
		self.history_out = []
		self.true_out = []
		self.false_out = []
		with tf.variable_scope("sentnce_rnn") as scope:
			for i in range(self.max_dialog_size):
				if self.use_lstm:
					cell = tf.nn.rnn_cell.LSTMCell(self.memory_size)
				else:
					cell = tf.nn.rnn_cell.GRUCell(self.memory_size)
				(outputs,_) = tf.nn.rnn(cell,self.history_emd[i],dtype=tf.float32) #创建一个rnn来执行
				self.history_out.append(outputs[-1])
				if i==0:    #第一次执行结束之后，开启重用变量的功能
					 tf.get_variable_scope().reuse_variables()

		with tf.variable_scope("dialog_rnn") as scope:
			if self.use_lstm:
				cell = tf.nn.rnn_cell.LSTMCell(self.memory_size)
			else:
				cell = tf.nn.rnn_cell.GRUCell(self.memory_size)
			(outputs,_) = tf.nn.rnn(cell,self.history_out,dtype=tf.float32) #创建一个rnn来执行
			self.context_out = outputs	#context rnn的历史时刻的输出
			self.concat_state = []	#进行状态合并之后待匹配的句子表达
			self.merge_weight = tf.Variable(tf.random_uniform([self.memory_size*2,self.memory_size],-1.0,1.0),name = 'merge')
			#进行状态合并，维度重新映射的权值矩阵

			for i in range(len(context_out)):
				if i%2==0:	#只有需要进行预测的时候进行状态合并
					concat_state = tf.concat(1,[self.context_outputs[i*2],self.encoder_outputs[i*2]])

		with tf.variable_scope("sentnce_rnn") as scope:
			#对sentence进行编码的rnn应当共享变量
			tf.get_variable_scope().reuse_variables()
			for i in range(len(self.true_index)):
				#遍历所有的候选对话
				if self.use_lstm:
					cell = tf.nn.rnn_cell.LSTMCell(self.memory_size)
				else:
					cell = tf.nn.rnn_cell.GRUCell(self.memory_size)
				(outputs,_) = tf.nn.rnn(cell,self.true_emd[i],dtype=tf.float32) #创建一个rnn来执行
				self.true_out.append(outputs[-1])	#获取最后一时刻的输出
			for i in range(len(self.false_index)):
				#遍历所有的候选对话
				if self.use_lstm:
					cell = tf.nn.rnn_cell.LSTMCell(self.memory_size)
				else:
					cell = tf.nn.rnn_cell.GRUCell(self.memory_size)
				(outputs,_) = tf.nn.rnn(cell,self.false_emd[i],dtype=tf.float32) #创建一个rnn来执行
				self.false_out.append(outputs[-1])	#获取最后一时刻的输出
			
				
############################################################################
			for i in range(self.max_dialog_size):
				if self.use_lstm:
					cell = tf.nn.rnn_cell.LSTMCell(self.memory_size)
				else:
					cell = tf.nn.rnn_cell.GRUCell(self.memory_size)

				(outputs,_) = tf.nn.rnn(cell,self.history_emd[i],dtype=tf.float32) #创建一个rnn来执行
				self.history_out.append(outputs[-1])
				if i==0:    #第一次执行结束之后，开启重用变量的功能
					
	def build_demo_model(self):
		
					 tf.get_variable_scope().reuse_variables()
