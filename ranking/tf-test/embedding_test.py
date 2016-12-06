#coding=utf-8
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
	memory_size = 10	#LSTM Cell维度
	vocab_size = 1000	#字典中包含多少单词
	embedding_size = 30	#词向量长度
	index_inputs =[]
	embedding_inputs= []
	embedding_weight = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name = "embedding_weight")
	
	for i in range(10):
		index= tf.placeholder(tf.int32,shape=[None],name= "index_{0}".format(i))
		index_inputs.append(index)
		embedding = tf.nn.embedding_lookup(embedding_weight,index)
		embedding_inputs.append(embedding)
		
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		for i in range(10000):
			input_feed = {}
			for i in range(10):
				input_feed[index_inputs[i].name] = np.array([[1,2],[2,1]])
		
			sess.run(init)
			print(sess.run(embedding_inputs,feed_dict=input_feed))
		
