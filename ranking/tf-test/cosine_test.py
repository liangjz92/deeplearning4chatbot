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
	for i in range(3):
		index= tf.placeholder(tf.int32,shape=[None],name= "index_{0}".format(i))
		index_inputs.append(index)
		embedding = tf.nn.embedding_lookup(embedding_weight,index)
		embedding_inputs.append(embedding)

	norm_a = tf.sqrt(tf.reduce_sum(tf.mul(embedding_inputs[0],embedding_inputs[0]),1))
	norm_b = tf.sqrt(tf.reduce_sum(tf.mul(embedding_inputs[1],embedding_inputs[1]),1))
	norm_c = tf.sqrt(tf.reduce_sum(tf.mul(embedding_inputs[2],embedding_inputs[2]),1))
	mul_ab =  tf.reduce_sum(tf.mul(embedding_inputs[0],embedding_inputs[1]),1)
	mul_ac =  tf.reduce_sum(tf.mul(embedding_inputs[0],embedding_inputs[2]),1)
	cos_ab = tf.div(mul_ab,tf.mul(norm_a,norm_b))
	cos_ac = tf.div(mul_ac,tf.mul(norm_a,norm_c))
	diff = cos_ab-cos_ac
	zero = tf.constant(0,shape=[2],dtype=tf.float32)
	margin = tf.constant(0.05,shape=[2],dtype=tf.float32)

	loss= tf.maximum(zero,tf.sub(margin,diff))
		
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		output_feed =[
			embedding_inputs[0],
			diff
			]
		input_feed = {}
		for i in range(3):
			input_feed[index_inputs[i].name] = np.array([1,2,3])
		
		sess.run(init)
		print(sess.run(output_feed,feed_dict=input_feed))
