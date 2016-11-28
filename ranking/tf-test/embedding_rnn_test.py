#coding=utf-8
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
	memory_size = 10	#LSTM Cell维度
	vocab_size = 1000	#字典中包含多少单词
	embedding_size = 300	#词向量长度
	encoder_inputs =[]
	embedding_weight = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name = "embedding_weight")

	for i in range(layer_size):
		cache=[]
		for j in range(3):
			cache.append(tf.placeholder(tf.int32,shape=[None],name = "in_{0}_{1}".format(i,j)))
		inputs.append(cache)
			
	print(inputs)
	outputs=[]
	with tf.variable_scope('encoders'):
		###定义共享的encoder cell
		for i in range(layer_size):
			cell = tf.nn.rnn_cell.GRUCell(memory_size)
			cell = tf.nn.rnn_cell.EmbeddingWrapper(cell, vocab_size, embedding_size)
			(output,_) = tf.nn.rnn(cell,inputs[i],dtype=tf.float32)
			outputs.append(output)
			if i==0:
				tf.get_variable_scope().reuse_variables()
		
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		input_feed = {}
		for i in range(layer_size):
			for j in range(3):
				input_feed[inputs[i][j].name] = np.array([1,2])
		print(np.array([1,2]))
		sess.run(init)
		print(sess.run(outputs,feed_dict=input_feed))
