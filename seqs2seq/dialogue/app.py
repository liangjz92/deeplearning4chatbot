#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import hred

model_dir ='./model_history/'	#保存历史的模型运行结果
train_path = './data/train.txt'
dev_path = './data/dev.txt'
test_path = './data/test.txt'
max_size= 1000
steps_per_checkpoint = 50
def read_data(source_path,max_size=10000):
	#读取数据，载入内存
	data_set=[]
	with tf.gfile.GFile(source_path, mode="r") as source_file:
		source = source_file.readline()
		counter = 0
		while source and (not max_size or counter < max_size):
			counter += 1
			if counter % 100000 == 0:
				print("  reading data line %d" % counter)
				sys.stdout.flush()
#print(source)
			source = source.decode('utf-8')
			source = source.strip()
			sentence_array = source.split('\t')
			cache =[]
			for sentence in sentence_array:
				sentence = sentence.strip()
				source_ids = [int(x) for x in sentence.split()]
				cache.append(source_ids)
			data_set.append(cache)
			
			source= source_file.readline()
	return data_set

def create_model(session,train):
	
	#传入session和是否为训练模型
	model = hred.HRED()
	model.build_model()
	model.train= train
	#ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	ckpt = tf.train.get_checkpoint_state(model_dir)
	if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())
	return model
def train():
	with tf.Session() as sess:
		model = create_model(sess, True)
		# Read data into buckets and compute their sizes.
		dev_set = read_data(dev_path)
		train_set = read_data(train_path, max_size)

		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		step_count = 0
		while True:
			# Get a batch and make a step.
			start_time = time.time()
			encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set,True)
			#提取数据
#_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, True)
			step_loss = model.step(sess, encoder_inputs, decoder_inputs, target_weights, True)
			print("step_loss",step_loss)
			#print ("step count",step_count)
			step_count+=1
			
			step_time += (time.time() - start_time) / steps_per_checkpoint
			
			#step_loss_sum= tf.reduce_sum(step_loss)
			#loss += step_loss_sum / steps_per_checkpoint
			#print("step_loss_sum",sess.run(step_loss_sum))
#			print("step_time",step_time)
			current_step += 1
		'''
		if current_step % steps_per_checkpoint == 0:	#测试
			perplexity = math.exp(loss) if loss < 300000 else float('inf')
			print ("global step %d learning rate %.4f step-time %.2f perplexity "
						               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
										                            step_time, perplexity))
			if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
				sess.run(model.learning_rate_decay_op)
			previous_losses.append(loss)
			checkpoint_path = os.path.join(model_dir, "hred.ckpt")
			model.saver.save(sess, checkpoint_path, global_step=model.global_step)
			'''
if __name__ == '__main__':
	train()
	
	
