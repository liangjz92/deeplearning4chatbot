#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import tensorflow as tf
import numpy as np
import data_utils
class HRED:
	def __init__(self):
		self.train = True	#模型用来训练还是测试
		self.batch_size = 2
		self.memory_size =10	#RNN单元维度
		self.vocab_size = 20000
		self.embedding_size = 30
		self.max_dialog_size = 3	#最长50次交互
		self.max_sentence_size = 4	#每句话长度为100个单词
		self.num_samples = 20000	#带采样的softmax
		self.learning_rate = tf.Variable(float(0.5),trainable =False,dtype= tf.float32)
		self.learning_rate_decay_factor = 0.99
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)
		self.max_gradient_norm =1.0
		self.test_set_index = 0	#测试集数据当前读取的进度
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
				target = [self.decoders[index][j] for j in xrange(self.max_sentence_size)]	#targets是decoder input向后顺延一位的结果
				self.targets.append(target)
		
		###################################################
		output_projection = None
		softmax_loss_function = None
		if self.num_samples >0 and self.num_samples <self.vocab_size:
			w_t =tf.get_variable('proj_w',[self.vocab_size,self.memory_size],dtype=tf.float32)
			w =tf.transpose(w_t)
			b =tf.get_variable('proj_b',[self.vocab_size],dtype=tf.float32)
			output_projection = (w,b)
			print(output_projection)
			
			def sampled_loss(inputs,labels):
				labels = tf.reshape(labels,[-1,1])
				print("labels",labels)
				return tf.nn.sampled_soft_max_loss(w_t,b,inputs,labels,self.num_samples,self.vocab_size)
				
			softmax_loss_function =  sampled_loss
		################################################################

		self.encoder_outputs = []	#各个编码器在最后一时刻的输出结果
		self.context_outputs = None	#上下文编码器的数据结果
		self.decoder_outputs = []	#各个解码器的解码结果，用以计算损失函数
		self.temp_outputs = []
		with tf.variable_scope("encoder") as scope:
			for i in range(self.max_dialog_size):
				cell = tf.nn.rnn_cell.GRUCell(self.memory_size)
				cell = tf.nn.rnn_cell.EmbeddingWrapper(cell,self.vocab_size,self.embedding_size)
				(outputs,_) = tf.nn.rnn(cell,self.encoders[i],dtype=tf.float32)	#创建一个rnn来执行
				self.temp_outputs.append(outputs)
				self.encoder_outputs.append(outputs[-1])	#提取最后一时刻的输出进行保存
#				self.encoder_outputs.append(outputs[0])	#提取最后一时刻的输出进行保存

				if i==0:	#第一次执行结束之后，开启重用变量的功能
					tf.get_variable_scope().reuse_variables()

		with tf.variable_scope("context") as scope:
			cell = tf.nn.rnn_cell.GRUCell(self.memory_size)	#用来编码上下文的rnn cell,没有任何特别之处
			(self.context_outputs,_) = tf.nn.rnn(cell,self.encoder_outputs,dtype=tf.float32)	#直接在encoder的结果上进行编码

		with tf.variable_scope("decoder") as scope:
			def argmax_loop_function(prev,_):
				if output_projection is not None:
				      prev = nn_ops.xw_plus_b(
							            prev, output_projection[0], output_projection[1])
				prev_symbol = math_ops.argmax(prev, 1)
				return prev_symbol

			for i in range(int(self.max_dialog_size/2)):
				cell = tf.nn.rnn_cell.GRUCell(self.memory_size)
				cell = tf.nn.rnn_cell.EmbeddingWrapper(cell,self.vocab_size,self.embedding_size)
				decoder_output = None
				if self.train:	#在训练过程中的输入是预先定义好的，无需处理
					(decoder_output,_) = tf.nn.seq2seq.rnn_decoder(self.decoders[i],self.context_outputs[i],cell,loop_function=None)
				else:	#在测试过程中的输入是动态确定的(第一步除外,所以需要输入)
					(decoder_output,_) = tf.nn.seq2seq.rnn_decoder(self.decoders[i],self.context_outputs[i],cell,loop_function = argmax_loop_function)
					if output_projection is not None:
						decoder_output = [ tf.matmul(output, output_projection[0]) + output_projection[1] for output in decoder_output]
														        
				self.decoder_outputs.append(decoder_output)
				if i==0:
					tf.get_variable_scope().reuse_variables()

		###################
		self.losses = []
		for i in range(int(self.max_dialog_size/2)):
			print(len(self.decoder_outputs[i]))
			print(len(self.targets[i]))
			print(len(self.weights[i]))
			print(softmax_loss_function)
			loss = tf.nn.seq2seq.sequence_loss(
					self.decoder_outputs[i],	#rnn解码器输出的结果，[ batch*symbol , ...]
					self.targets[i],	#预期的单词		[batch, ....]
					self.weights[i],	#计算loss时的权重 [batch, ...]
					softmax_loss_function
					)
			self.losses.append(loss)
		self.loss_sum = tf.reduce_sum(tf.pack(self.losses))	#final loss
		
		self.gradient_norms = []
		self.updates=[]
		opt = tf.train.GradientDescentOptimizer(self.learning_rate)
		params = tf.trainable_variables()
		gradients = tf.gradients(self.loss_sum,params)
		clipped_gradients, norm = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
		self.gradient_norms.append(norm)
		self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
		self.saver = tf.train.Saver(tf.all_variables())
			
	def step(self,session,encoder_inputs,decoder_inputs,weight_inputs,train):
		'''
		print("step()")
		print("encoder_inputs",encoder_inputs)
		print("decoder_inputs",decoder_inputs)
		print("weight_inputs",weight_inputs)
		print("train",train)
		'''
		#喂一个batch的数据给模型，让模型单独跑一步,模型已经在初始化的时候生成结束
		input_feed={}	#参数映射
		for i in range(self.max_dialog_size):	#多句对话
			for j in range(self.max_sentence_size):	#每句对话多个单词(后面还有batch维度)
				input_feed[self.encoders[i][j].name] = encoder_inputs[i][j]
#				print(self.encoders[i][j])
#				print(encoder_inputs[i][j])
		for i in range(int(self.max_dialog_size/2)):
			for j in range(self.max_sentence_size):
				input_feed[self.decoders[i][j].name] = decoder_inputs[i][j]
				input_feed[self.weights[i][j].name] = weight_inputs[i][j]

		if train:
			output_feed=[
#				self.updates,
#self.gradient_norms,
				self.losses[0]
			]
		else:
			output_feed =[self.loss_sum]
			for i in range(int(self.max_dialog_size)):
				for j in range(self.max_sentence_size):
					output_feed.append(self.outputs[i][j])
		for item in output_feed:
			pass
			#print(item)
#		print(output_feed)
#		print(input_feed)
#		print(outputs)
#		print("outputs done")
#		print ("losses",session.run(self.losses,input_feed))
#		print ("temp_outputs",session.run(self.temp_outputs,input_feed))
#		print ("encoders",session.run(self.encoders,input_feed))
#		print ("encoder_outputs",session.run(self.encoder_outputs,input_feed))
#		print ("decoder_outputs",session.run(self.decoder_outputs,input_feed))
#		print ("context_outputs",session.run(self.context_outputs,input_feed))
		outputs = session.run(output_feed, input_feed)
		print("loss",outputs[0])
		return outputs
	'''
		if train:
			return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
		else:
			return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
	'''			
	def get_batch(self,data,train=True):	# 一个数据集、是否是训练集，训练集随机取，测试集挨个遍历
		dialogs= []		#暂存选出的样本
		if train ==True:	#训练集随机取batch个样本
			for i in xrange(self.batch_size):
				dialogs.append(random.choice(data))
			return self.assemble(dialogs)
		else:	#测试集按顺序进行选取
			if self.test_set_index > len(data):	#测试已经取完
				return assemble(None)
			test_size = min(self.batch_size,len(data)-self.test_set_index)	#这次应该有多少个样本
			for i in range(test_size):
				dialogs.append(data[self.test_set_index])
				self.test_set_index +=1
			return self.assemble(dialogs)			
			
	def assemble(self,dialogs):	#给定若干个对方，返回他们的向量表示，用作模型输入
		if dialogs ==None:	#没传进来参数
			return None,None,None
		encoder_inputs=[]
		decoder_inputs=[]
		batch_size = len(dialogs)
		for i in range(batch_size):	#batch
			one_session = dialogs[i]	#对话真实存在
			cache = []
			for j in range(self.max_dialog_size):	#句子可能因为对话长度不够而不存在
				if j< len(one_session):
					encoder_pad = [data_utils.PAD_ID]*(self.max_sentence_size-len(one_session[j]))
					cache.append(list(reversed(one_session[j]+encoder_pad)))
				else:
					cache.append(list([data_utils.PAD_ID]*self.max_sentence_size))
			encoder_inputs.append(cache)
			
			cache = []
			for j in range(self.max_dialog_size):	#decoder部分
				if j %2==0:	#第0,2,4,..句话由用户说
					continue
				if j<len(one_session):
					decoder_pad = [data_utils.PAD_ID]*(self.max_sentence_size-len(one_session[j])-2)
					cache.append([data_utils.GO_ID]+one_session[j]+[data_utils.EOS_ID]+decoder_pad)
				else:
					cache.append([data_utils.GO_ID]+[data_utils.PAD_ID]*(self.max_sentence_size-1))
			decoder_inputs.append(cache)
		
		########################
		
		batch_encoder,batch_decoder,batch_weight =[],[],[]
		for sent_index in range(self.max_dialog_size):
			cache = []
			for length_index in range(self.max_sentence_size):
				cache.append(np.array([encoder_inputs[batch_index][sent_index][length_index]for batch_index in range(len(encoder_inputs))]))
			batch_encoder.append(cache)
#		print("len(decoder_inputs)",len(decoder_inputs))
		for sent_index in range(len(decoder_inputs[0])):	#遍历deoder的输入
			#输入的decoder input最外层是各个不用样本，所以这里取0好样本
			cache = []	#decoder_cache
			weight_cache=[]	#weight_cache
			for length_index in range(self.max_sentence_size):	#token size
				cache.append(np.array([decoder_inputs[batch_index][sent_index][length_index]for batch_index in range(len(decoder_inputs))]))
				single_weight=np.ones(len(decoder_inputs),dtype=np.float32)
				for i in range( len(decoder_inputs)):	#batch size
					if length_index == self.max_sentence_size-1 or cache[-1][i]==data_utils.PAD_ID:
						single_weight[i]= 0.0
				weight_cache.append(single_weight)
						
			batch_decoder.append(cache)
			batch_weight.append(weight_cache)
		return batch_encoder,batch_decoder,batch_weight	
		


				
		
										
if __name__ == '__main__':
	model = HRED()
	init = tf.initialize_all_variables()	
	with tf.Session() as sess:
		sess.run(init)
		model.build_model()
#		sess.run(data,feed_dict={data:np.array([[[1.0,2.0]]])})
#		print(sess.run(output,feed_dict={data:np.array([[[1.0,2.0]]])}))