#coding=utf-8
import tensorflow as tf
import numpy as np
if __name__ =='__main__':
	a=tf.placeholder(tf.float32,[1,2])
	b=tf.placeholder(tf.float32,[1,2])
	with tf.variable_scope('abc') as scope:

		cell_1= tf.nn.rnn_cell.GRUCell(2)
		cell_2= tf.nn.rnn_cell.GRUCell(2)
		h,w= cell_1(a,b,scope='abc')
		#tf.get_variable_scope().reuse_variables()
		h2,w= cell_2(a,b,scope='abd')
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		in_dict={a:np.array([[2.0,3.0]]),b:np.array([[1.0,2.0]])}
		print(sess.run(h,feed_dict=in_dict))
		print(sess.run(h2,feed_dict=in_dict))
		
#print(sess.run(cell_1))
	print("done")
