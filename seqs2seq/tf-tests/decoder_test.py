#coding=utf-8
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
	embedding_size =5
	memory_size = 15
	vocab_size = 20
	with tf.variable_scope('encoder'):
		#embedding = tf.placeholder(tf.float32,[vocab_size,embedding_size])
		embedding = tf.get_variable('embedding_w',[vocab_size,embedding_size],dtype=tf.float32)
		ids = tf.placeholder(tf.int32,[2,3])
		emb = tf.nn.embedding_lookup(embedding,ids)
		cell = tf.nn.rnn_cell.GRUCell(memory_size)
		init_state = tf.get_variable('init_state',[2,15],dtype=tf.float32)	#初始的状态
		
		output,state = tf.nn.dynamic_rnn(cell,emb,dtype=tf.float32)
		
		'''
		data = tf.placeholder(tf.float32,[None,None,2])
		cell = tf.nn.rnn_cell.GRUCell(2)
		cell = tf.nn.rnn_cell.
		output,state=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)'''
		init = tf.initialize_all_variables()	
	with tf.Session() as sess:
		sess.run(init)
		print(sess.run(emb,feed_dict={ids:np.array([[1,2,3],[6,7,8]])}))
		print(sess.run(output,feed_dict={ids:np.array([[1,2,3],[6,7,8]])}))
