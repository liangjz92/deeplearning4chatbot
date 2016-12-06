#coding=utf-8
import tensorflow as tf
import numpy as np
from data_utils import DU
if __name__ == '__main__':
	du = DU()
	vocab ,recab = du.initialize_vocabulary()
	vocab_size =len(vocab)
	embedding_size = 300
	emd = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
	weight = np.random.rand(vocab_size,embedding_size)*2-1
	f = open('../data/emd/cyemd.bin','r')
	first = True
	for line in f:
		if first:
			first=False
			continue
		box = line.split(' ')
		word = box[0]
		box= box[1:301]
		if vocab.has_key(word):
			index = vocab[word]
			print(word,index)
			#print(box)
			one_emd = np.array([float(x) for x in box])
			#print(one_emd)
			weight[index,:] = one_emd
	load = emd.assign(weight)
	with tf.Session() as sess:
		sess.run(load)


		

