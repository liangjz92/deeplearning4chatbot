#coding=utf-8
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
	memory_size = 100	#LSTM Cell维度
	vocab_size = 10000	#字典中包含多少单词
	embedding_size = 300	#词向量长度
	
	dialog_data = []
	for i in range(50):
		dialog_data.append(tf.placeholder(tf.float32,[None,None,1]))
	dialog_mark=tf.placeholder(tf.float32,[50])

	with tf.variable_scope('encoders'):
		###定义共享的encoder cell
		encoder_cell = tf.nn.rnn_cell.GRUCell(memory_size)
		encoder_cell = tf.nn.rnn_cell.EmbeddingWrapper(encoder_cell, vocab_size, embedding_size)
		encoder_outputs_array=[]
		encoder_outputs,encoder_state = tf.nn.dynamic_rnn(encoder_cell,dialog_data[0],sequence_length=None,dtype=tf.float32)
		encoder_outputs_array.append(encoder_outputs)
		tf.get_variable_scope().reuse_variables()
		for i in range(1,50):
			encoder_outputs,encoder_state = tf.nn.dynamic_rnn(encoder_cell,dialog_data[i])
			encoder_outputs_array.append(encoder_outputs)
		
	with tf.variable_scope('context'):
		context_cell = tf.nn.rnn_cell.GRUCell(memory_size)
		
	with tf.variable_scope('decoders'):
		decoder_cell = tf.nn.rnn_cell.GRUCell(memory_size)
		decoder_cell = tf.nn.rnn_cell.GRUCell(decoder_cell,vocab_size,embedding_size)
	
		
		
		
		
	init = tf.initialize_all_variables()	
	with tf.Session() as sess:
		sess.run(init)
#		sess.run(data,feed_dict={data:np.array([[[1.0,2.0]]])})
#		print(sess.run(output,feed_dict={data:np.array([[[1.0,2.0]]])}))
#		print(sess.run(output1,feed_dict={data:np.array([[[1.0,2.0],[2.0,1.0]]])}))
#		print(sess.run(output1,feed_dict={data:np.array([[[1.0,2.0]]])}))
		
