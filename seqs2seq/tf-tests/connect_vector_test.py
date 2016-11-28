#coding=utf-8
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
	a = tf.placeholder(tf.float32,[None])
	b = tf.placeholder(tf.float32,[None])
	c = tf.concat(0,[a,b])
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		feed_dict = {}
		feed_dict[a.name]  = np.array([1,2])
		feed_dict[b.name]  = np.array([3,4])
		sess.run(init)
		print(sess.run(c,feed_dict))
		'''
		sess.run(data,feed_dict={data:np.array([[[1.0,2.0]]])})
		print(sess.run(output,feed_dict={data:np.array([[[1.0,2.0]]])}))
		print(sess.run(output1,feed_dict={data:np.array([[[1.0,2.0],[2.0,1.0]]])}))
		print(sess.run(output1,feed_dict={data:np.array([[[1.0,2.0]]])}))
	'''
		
