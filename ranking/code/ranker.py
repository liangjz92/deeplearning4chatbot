#coding=utf-8
import tensorflow as tf
import numpy as np
import data_utils
class Ranker:
	def __init__(self,
			vocab_size = 20001,
			embedding_size = 10,
			memory_size = 10,
			batch_size =20,
			max_dialogue_size = 5,
			max_sentence_size = 6,
			l2_weight = 1e-7,
			margin = 0.05,
			max_gradient_norm = 5.0,
			learning_rate =1.0,
			learning_rate_decay_factor = 0.9,
			use_lstm = False,
			train_mode = True
			):
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size 
		self.memory_size= memory_size
		self.batch_size = batch_size
		self.max_dialogue_size = max_dialogue_size
		self.max_sentence_size = max_sentence_size
		self.l2_weight = l2_weight
		self.max_gradient_norm = max_gradient_norm
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
		self.use_lstm = use_lstm 
		self.train_mode =train_mode
		self.global_step = tf.Variable(0, trainable=False)
		self.margin_val = margin

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
			for j in range(self.max_sentence_size):
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
			for j in range(self.max_sentence_size):
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
			for i in range(self.max_dialogue_size):
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
			for i in range(len(self.context_out)):
				if i%2==0:	#只有需要进行预测的时候进行状态合并
					#concat_state = self.context_out[i]
					concat_state = tf.concat(1,[self.context_out[i],self.history_out[i]])
					concat_state = tf.matmul(concat_state,self.merge_weight)
					self.concat_state.append(concat_state)
			#合并完成
					
		with tf.variable_scope("sentnce_rnn") as scope:
			#对sentence进行编码的rnn应当共享变量
			tf.get_variable_scope().reuse_variables()
			for i in range(len(self.true_index)):
				#遍历所有的候选对话.true candiate
				#tf.get_variable_scope().reuse_variables()
				cell = None
				if self.use_lstm:
					cell = tf.nn.rnn_cell.LSTMCell(self.memory_size)
				else:
					cell = tf.nn.rnn_cell.GRUCell(self.memory_size)
				(outputs,_) = tf.nn.rnn(cell,self.true_emd[i],dtype=tf.float32) #创建一个rnn来执行
				self.true_out.append(outputs[-1])	#获取最后一时刻的输出
			for i in range(len(self.false_index)):
				#遍历所有的候选对话,false candidate
				if self.use_lstm:
					cell = tf.nn.rnn_cell.LSTMCell(self.memory_size)
				else:
					cell = tf.nn.rnn_cell.GRUCell(self.memory_size)
				(outputs,_) = tf.nn.rnn(cell,self.false_emd[i],dtype=tf.float32) #创建一个rnn来执行
				self.false_out.append(outputs[-1])	#获取最后一时刻的输出
		
		with tf.variable_scope("loss_compute") as scope:
			self.true_norm, self.false_norm, self.state_norm =[], [], []
			self.mul_ts, self.mul_fs = [], []
			self.cos_ts, self.cos_fs = [], []
			self.loss = []
			self.zero = tf.placeholder(tf.float32,shape=[None],name="zero")
			self.margin = tf.placeholder(tf.float32,shape=[None],name="margin")
			
			for i in range(min(len(self.concat_state),len(self.false_out))):
				#遍历可用于计算loss的所有输出
				self.true_norm.append(tf.sqrt(tf.reduce_sum(tf.mul(self.true_out[i],self.true_out[i]),1)))	# norm of true
				self.false_norm.append(tf.sqrt(tf.reduce_sum(tf.mul(self.false_out[i],self.false_out[i]),1)))	#norm of false
				self.state_norm.append(tf.sqrt(tf.reduce_sum(tf.mul(self.concat_state[i],self.concat_state[i]),1)))	#norm of state
				self.mul_ts.append( tf.reduce_sum( tf.mul(self.true_out[i],self.concat_state[i] ),1))	#true&state mul
				self.mul_fs.append( tf.reduce_sum( tf.mul(self.false_out[i],self.concat_state[i] ),1))	#false&state mul
				self.cos_ts.append( tf.div( self.mul_ts[i], tf.mul( self.true_norm[i], self.state_norm[i])))	#true&state cosine
				self.cos_fs.append( tf.div( self.mul_fs[i], tf.mul( self.false_norm[i], self.state_norm[i])))	#false&state cosine
				self.loss.append( tf.maximum( self.zero, tf.sub( self.margin, tf.sub( self.cos_ts[i], self.cos_fs[i]))))
				#margin loss max (0,margin-diff)  diff= cos_ts-cos_fs

		with tf.variable_scope("loss_compute") as scope:
			#在训练过程中计算损失函数
			self.loss_mean = tf.reduce_sum(self.loss)
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
			opt = tf.train.AdamOptimizer(self.learning_rate)
			gradients = tf.gradients(self.loss_mean,params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
			self.gradient_norms.append(norm)
			self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
		
		with tf.variable_scope('cosine_test'):
			#在测试过程中批量计算cosine的值
			self.test_context = tf.placeholder(tf.float32,shape=[None,self.memory_size],name="test_context")
			self.test_candidate = tf.placeholder(tf.float32,shape=[None,self.memory_size],name="test_candidate")
			self.context_norm = tf.sqrt(tf.reduce_sum(tf.mul( self.test_context, self.test_context),1))	# norm of context
			self.candidate_norm = tf.sqrt(tf.reduce_sum(tf.mul( self.test_candidate, self.test_candidate),1))	# norm of candiate
			self.mul_cc = tf.reduce_sum( tf.mul(self.test_context,self.test_candidate ),1)	#cc mul
			self.cos_cc = tf.div( self.mul_cc, tf.mul(self.context_norm, self.candidate_norm))	#cc cosine
			
		self.saver = tf.train.Saver(tf.all_variables())
		#保存所有模型参数信息

	def step_train(self,session,history,true_index,false_index):
		#进行一次训练迭代
		input_feed={}	
		for i in range(self.max_dialogue_size):
			for j in range(self.max_sentence_size):
				input_feed[self.history_index[i][j].name] = history[i][j]	

		for i in range(len(self.true_index)):
			for j in range(self.max_sentence_size):
				input_feed[self.true_index[i][j].name] = true_index[i][j]
				input_feed[self.false_index[i][j].name] = false_index[i][j]

		input_feed[self.zero.name] = np.array([0.0 for i in xrange(history[0][0].shape[0])])
		input_feed[self.margin.name] = np.array([self.margin_val for i in xrange(history[0][0].shape[0])])
		output_feed = [
			self.updates,
		    self.gradient_norms,
		    self.loss_mean,
#self.loss,
			#self.true_out

		]
		outputs = session.run(output_feed,input_feed)
#		print('self.loss',outputs[3])
		return outputs[2]

	def step_test(self,session,history,candidates):
		#执行一步测试
		input_feed ={}
		for i in range(self.max_dialogue_size):
			for j in range(self.max_sentence_size):
				input_feed[self.history_index[i][j].name] = history[i][j]
		#填充历史信息

		for i in range(len(self.true_index)):
			for j in range(self.max_sentence_size):
				input_feed[self.true_index[i][j].name] = candidates[i][j]
		output_feed = [
		  self.concat_state,
		  self.true_out
		]
		outputs = session.run(output_feed,input_feed)
		
		###########
		response_size = len(outputs[1])	#一次对话中，需要进行多少次预测
		candidate_size = len(outputs[1][0])	#每次预测，需要候选多少个样本
		#print("reponse_size",response_size)
		#print("candiate_size",candidate_size)
		results = []
		for i in range(response_size):
			context = outputs[0][i]
			context = np.repeat(context,candidate_size,0)
			input_feed = {}
			input_feed[self.test_context.name] =  context
			input_feed[self.test_candidate.name] = outputs[1][i]
			output_feed = [self.cos_cc]
			output= session.run(output_feed,input_feed)
			results.append(output[0])
			#print("cosine outputs",output)
		return results

############################################################################
	def get_max(self, iters):
		#获取当前迭代次数下，有效的对话长度限制
		#课程学习
		return 100
		border = 1
		for i in range(1,100):
			if i*1000+i*i*200 < iters:
				border = i
			else:
				break
		return border

#################################

	def train2vec(self, dialogs, iters):
		batch_size = len(dialogs)	#获取当前batch_size
		max_border = self.get_max(iters)	#当前最多可以说几句话
		history_inputs =[]
		true_inputs =[]
		false_inputs = []
		for i in range( batch_size ):
			border = min(len(dialogs[i]),max_border*2)
			dialogs[i] = dialogs[i][:border]
			#for j in len(dialogs[i]):
		if (dialogs ==None) or len(dialogs)==0 : #没传进来参数
			return None,None,None
		for i in range(batch_size): #batch
			one_session = dialogs[i]	#对话真实存在
			cache = []
			for j in range(self.max_dialogue_size):	#句子可能因为对话长度不够而不存在
				if j < len(one_session):
					encoder_pad = [data_utils.PAD_ID]*(self.max_sentence_size-len(one_session[j][0]))	#0是真实的对话
					#print('encoder_pad',encoder_pad)
					cache.append(list(reversed(one_session[j][0]+encoder_pad)))	#反转输入
				else:
					cache.append(list([data_utils.PAD_ID]*self.max_sentence_size))
			history_inputs.append(cache)
			true_cache =[]
			false_cache = []
			for j in range(self.max_dialogue_size):   #candidate part
				if j %2==0: #第0,2,4,..句话由用户说
					continue
				if j<len(one_session):
					true_pad = [data_utils.PAD_ID]*(self.max_sentence_size-len(one_session[j][0]))
					true_cache.append(list(reversed(one_session[j][0] + true_pad)))# true candiate
					false_pad = [data_utils.PAD_ID]*(self.max_sentence_size-len(one_session[j][1]))
					false_cache.append(list(reversed(one_session[j][1] + false_pad)))#false candidate
				else:
					true_cache.append(list([data_utils.EOS_ID]*self.max_sentence_size))
					false_cache.append(list([data_utils.PAD_ID]*self.max_sentence_size))
#			print('true_cache',true_cache)
#			print('false_cache',false_cache)
			true_inputs.append(true_cache)
			false_inputs.append(false_cache)
		#print('true_inputs',true_inputs)
		#print('false_inputs',false_inputs)
		######################################################
		batch_history,batch_true,batch_false = [], [], []
		for sent_index in range(self.max_dialogue_size):
			history_cache = []
			for length_index in range(self.max_sentence_size):
				history_cache.append(np.array([history_inputs[batch_index][sent_index][length_index] for batch_index in range(len(history_inputs))]))
			batch_history.append(history_cache)
			if sent_index % 2!=0:
				true_cache, false_cache = [], []
				for length_index in range(self.max_sentence_size):
					true_cache.append(np.array([true_inputs[batch_index][int(sent_index/2)][length_index] for batch_index in range(len(history_inputs))]))
					false_cache.append(np.array([false_inputs[batch_index][int(sent_index/2)][length_index] for batch_index in range(len(history_inputs))]))
				batch_true.append(true_cache)
				batch_false.append(false_cache)

		return batch_history, batch_true, batch_false			
		
	def test2vec(self,history):
		#将测试数据转换成合适的格式
		#测试数据每次只使用一条
		history_inputs =[]
		candidate_inputs =[]
		if (history ==None) or len(history)==0 : #没传进来参数
			return None,None
		candidate_size = len(history[1])
		#print('candidate_size',candidate_size)
		cache = []
		for j in range(self.max_dialogue_size):	#句子可能因为对话长度不够而不存在
			if j< len(history):
				encoder_pad = [data_utils.PAD_ID]*(self.max_sentence_size-len(history[j][0]))	#0是真实的对话
				cache.append(list(reversed(history[j][0]+encoder_pad)))	#反转输入
			else:
				cache.append(list([data_utils.PAD_ID]*self.max_sentence_size))
		history_inputs = cache
		#print(history_inputs)
		true_cache =[]
		for i in range(self.max_dialogue_size):   #candidate part
			if i %2==0: #第0,2,4,..句话由用户说
				continue
			if i<len(history):	#这个对话是存在的
				for j in range(candidate_size):
					true_pad = [data_utils.PAD_ID]*(self.max_sentence_size-len(history[i][j]))
					true_cache.append(list(reversed(history[i][j] + true_pad)))# true candidate
			else:
				for j in range(candidate_size):
					true_cache.append(list([data_utils.PAD_ID]*self.max_sentence_size))
			candidate_inputs.append(true_cache)
			true_cache =[]
		
		
		######################################################
		batch_history, batch_candidate = [], []

		for sent_index in range(self.max_dialogue_size):
			history_cache = []
			for length_index in range(self.max_sentence_size):
				history_cache.append(np.array( [history_inputs[sent_index][length_index]]))
			batch_history.append(history_cache)

			if sent_index % 2 != 0:
				candidate_cache = []
				for length_index in range(self.max_sentence_size):
					candidate_cache.append(np.array([candidate_inputs[int(sent_index/2)][batch_index][length_index] for batch_index in range(candidate_size)]))
				batch_candidate.append(candidate_cache)
		return batch_history, batch_candidate

############################################################################
					
	def test(self):
		history = []
		for i in range(self.max_dialogue_size):
			temp =[]
			for j in range(self.max_sentence_size):
				temp.append(np.array([3]))
			history.append(temp)

		candidate = []
		for i in range(int(self.max_dialogue_size/2)):
			temp =[]
			for j in range(self.max_sentence_size):
				temp.append(np.array([3,4,5]))
			candidate.append(temp)
		with tf.Session() as sess:
			init = tf.initialize_all_variables()
			sess.run(init)
			self.step_test(sess,history,candidate)



if __name__ == '__main__':
	temp = Ranker()
	temp.build_model()
	temp.test()
		
