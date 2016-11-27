#coding=utf-8
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
	with tf.variable_scope('foo'):
		data = tf.placeholder(tf.float32,[None,None,2])
		cell = tf.nn.rnn_cell.GRUCell(2)
		output,state=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)
		tf.get_variable_scope().reuse_variables()
		output1,state1=tf.nn.dynamic_rnn(cell,data,dtype=tf.float32)
		init = tf.initialize_all_variables()	
	with tf.Session() as sess:
		sess.run(init)
		sess.run(data,feed_dict={data:np.array([[[1.0,2.0]]])})
		print(sess.run(output,feed_dict={data:np.array([[[1.0,2.0]]])}))
		print(sess.run(output1,feed_dict={data:np.array([[[1.0,2.0],[2.0,1.0]]])}))
		print(sess.run(output1,feed_dict={data:np.array([[[1.0,2.0]]])}))
		
