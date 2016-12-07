#coding=utf-8
import tensorflow as tf
import numpy as np
#import data_utils
#根据的描述产生标签，做成分类模型
class Marker:
	def __init__(self,
			vocab_size = 30001,
			embedding_size = 10,
			memory_size = 10,
			batch_size =20,		
			max_ut_size = 7,	#句子最长是多少
			lable_size = 5,	#总共有多少种标签
			l2_weight = 1e-7,
			max_gradient_norm = 5.0,
			learning_rate =0.01,
			learning_rate_decay_factor = 0.95,
			train_mode = True
			):

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size 
		self.memory_size= memory_size
		self.batch_size = batch_size
		self.max_ut_size = max_ut_size
		self.l2_weight = l2_weight
		self.max_gradient_norm = max_gradient_norm
		self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
		self.train_mode =train_mode
		self.global_step = tf.Variable(0, trainable=False)
		self.label_size  =lable_size

	def build_model(self):
		self.ut_index =[]
		self.ut_emd = []

		with tf.variable_scope('embedding') as scope:
			self.embedding_weight = tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0),name = 'embedding_weight')
			tf.histogram_summary('embedding_weight',self.embedding_weight)	#词向量权重的统计
			for i in range (self.max_ut_size):	#最长的句子
				self.ut_index.append(tf.placeholder(tf.int32,shape=[None],name="ut_index_{0}".format(i)))
				self.ut_emd.append(tf.nn.embedding_lookup(self.embedding_weight,self.ut_index[-1]))

		self.labels =tf.placeholder(tf.float32, shape = [ None,self.label_size], name = 'labels')
		
		########################################
		self.ut_out = []
		with tf.variable_scope("rnn") as scope:
			cell = tf.nn.rnn_cell.GRUCell(self.memory_size)
			(outputs,_) = tf.nn.rnn(cell,self.ut_emd,dtype=tf.float32) #创建一个rnn来执行
			self.ut_rep = outputs[-1]

		with tf.variable_scope('transform'):
			self.weight = tf.Variable(tf.random_uniform([self.memory_size, self.label_size],-1.0,1.0),name = 'weight')
			self.offset = tf.Variable(tf.random_uniform([self.label_size],-1.0,1.0),name = 'offset')
			self.logits = tf.add( tf.matmul(self.ut_rep,self.weight), self.offset)
		
		with tf.variable_scope("loss_compute") as scope:
			self.loss = tf.nn.softmax_cross_entropy_with_logits(self.logits,self.labels)

		with tf.variable_scope("train_op") as scope:
			#在训练过程中计算损失函数
			self.loss_mean = tf.reduce_mean(self.loss,0)
			self.loss_sum = tf.reduce_sum(self.loss)
			tf.scalar_summary('batch_reduce_mean_loss',self.loss_mean)
			tf.scalar_summary('batch_reduce_sum_loss',self.loss_sum)
			self.gradient_norms = []
			self.updates = []
			opt = tf.train.AdadeltaOptimizer(self.learning_rate)
			#优化器
			params = tf.trainable_variables()
			for param in params:
				if 'embedding' not in param.name:
					#print(param.name,param)
					self.loss_mean = self.loss_mean + self.l2_weight * tf.nn.l2_loss(param)
			#opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			#opt = tf.train.AdamOptimizer(self.learning_rate)
			gradients = tf.gradients(self.loss_mean,params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
			self.gradient_norms.append(norm)
			self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
		
		with tf.variable_scope('test'):
			#在测试过
			self.predicts = tf.nn.softmax(self.logits)
			
		self.saver = tf.train.Saver(tf.all_variables())
		self.merged = tf.merge_all_summaries()
		#保存所有模型参数信息
#####################################################

	def step_train(self,session,ut_arr,lable_arr):
		#进行一次训练迭代
		input_feed={}
		for i in range(self.max_ut_size):
			input_feed[self.ut_index[i].name] = ut_arr[i]
		input_feed [self.labels] = label_arr
		output_feed = [
			self.updates,
		    self.gradient_norms,
		    self.loss_mean,	#损失函数
			self.merged		#summary
		]
		outputs = session.run(output_feed,input_feed)
		#print(outputs[4])
		return outputs[2],outputs[3]

######################################################
	def step_test(self,session,ut_arr,label_arr):

		input_feed={}
		for i in range(self.max_ut_size):
			input_feed[self.ut_index[i].name] = ut_arr[i]
		input_feed [self.labels] = label_arr

		output_feed = [
		  self.predicts,
		  self.loss
		]
		outputs = session.run(output_feed,input_feed)
		return outputs

######################################################
	def step_demo(self, session,ut_arr):
		#根据输入预测标签
		input_feed={}
		for i in range(self.max_ut_size):
			input_feed[self.ut_index[i].name] = ut_arr[i]

		output_feed = [
		  self.predicts,
		]
		outputs = session.run(output_feed,input_feed)
		return outputs[0]
#######################################################
	def sample2vec(self, sample_arr):
		# 输入多个样本，每个样本0：ids  1：label ids
		ut_arr = []
		batch_size = len(sample_arr)
		labels = np.zeros((batch_size, self.label_size))
		vec_cahce = []
		for i in range(batch_size):
			pad = [data_utils.PAD_ID]*(self.max_ut_size-len(sample_arr[i][0]))	#0是句子
			vec_cache.append(list(reversed(sample_arr[i][0]+pad)))	#反转输入
			for j in range(len(sample_arr[i][1]))
				index = int (sample_arr[i][1][j])
				if index < self.label_size:
					labels[i][index] = 1.0
		for i in range(self.max_ut_size):
			temp = np.array([ vec_cache[index][i] for index in range(batch_size)])
			ut_arr.append(temp)
		#返回句子的id数组、lable的数组
		return ut_arr, labels
			
#######################################################
	def demo2vec(self,sentece):
		ut_arr = []
		batch_size = len(sample_arr)
		vec_cahce = []
		for i in range(batch_size):
			pad = [data_utils.PAD_ID]*(self.max_ut_size-len(sample_arr[i][0]))	#0是句子
			vec_cache.append(list(reversed(sample_arr[i][0]+pad)))	#反转输入
		for i in range(self.max_ut_size):
			temp = np.array([ vec_cache[index][i] for index in range(batch_size)])
			ut_arr.append(temp)
		return ut_arr
#######################################################

	def test(self):
		feed_dict = {}
		for i in range(self.max_ut_size):
			feed_dict[self.ut_index[i].name] =  np.array([3,4])
		
		labels = np.zeros((2,5))
		print(labels)
		for i in range(5):
			labels[i%2][i] = 1.0
		feed_dict[self.labels.name] = labels
		output_dict =[
			self.updates,
			self.gradient_norms,
			self.loss,
			self.logits,
			self.predicts
		]
		with tf.Session() as sess:
			init = tf.initialize_all_variables()
			sess.run(init,None)
			for i in range (10000):
				out = sess.run(output_dict,feed_dict)
				print(out[2],out[3],out[4])
			


if __name__ == '__main__':
	temp = Marker()
	temp.build_model()
	temp.test()
		
