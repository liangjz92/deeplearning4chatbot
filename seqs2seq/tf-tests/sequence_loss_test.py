#coding=utf-8
import tensorflow as tf
import numpy as np
import pdb
if __name__ == '__main__':
	memory_size = 10	#LSTM Cell维度
	vocab_size = 10	#字典中包含多少单词
	embedding_size = 300	#词向量长度
	logits = []
	targets = []
	weights = []
	sentence_length = 6
	batch_size = 2
	symbol_size = 3
	for i in range (sentence_length):
		logits.append(tf.placeholder(tf.float32,shape=[batch_size,symbol_size],name = "logits_{0}".format(i)))
		targets.append(tf.placeholder(tf.int32,shape=[batch_size],name = "targets_{0}".format(i)))
		weights.append(tf.placeholder(tf.float32,shape=[batch_size],name = "weights_{0}".format(i)))
	w_t = tf.get_variable("proj_w", [vocab_size, symbol_size], dtype=tf.float32)
	w = tf.transpose(w_t)
	b = tf.get_variable("proj_b", [vocab_size], dtype=tf.float32)
	op = (w, b)
	
	targets2 =tf.reshape(targets,[-1,1])
	logits2 = tf.reshape(logits,[-1,3])
	print("targets2",targets2)
	print("logits2",logits2)
	loss2 = tf.nn.sampled_softmax_loss(w_t,b,logits2,targets2,2,vocab_size)
	#pdb.set_trace()
	loss = tf.nn.seq2seq.sequence_loss(logits,targets,weights)
	
	#pdb.set_trace()
	init = tf.initialize_all_variables()	
		
	with tf.Session() as sess:
		input_feed = {}
		for i in range(sentence_length):
			input_feed[logits[i].name] = np.random.rand(batch_size,symbol_size)
			input_feed[targets[i].name] = np.random.randint(0,vocab_size-1,size=(batch_size))
			input_feed[weights[i].name] = np.ones(batch_size)
		sess.run(init)
		print(input_feed)
		print(sess.run(loss2,feed_dict=input_feed))
