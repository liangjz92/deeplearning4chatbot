#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from data_utils import DU
from ranker import Ranker
import json
########################################
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.98, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 100.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("margin", 0.05, "margin between true and false candiate")
tf.app.flags.DEFINE_integer("batch_size", 25, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("emd_size", 100, "embedding size")
tf.app.flags.DEFINE_integer("mem_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size", 30000, "vocabulary size.")
tf.app.flags.DEFINE_integer("max_dialogue_size", "10", "how manay uts in one sess max,25")
tf.app.flags.DEFINE_integer("max_sentence_size", "10", "how manay tokens in one sentence max 36")
tf.app.flags.DEFINE_integer("max_trainset_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer('max_devset_size',1000,"how many dev samples use max")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 500, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("train", True, "True to train model, False to decode model")
tf.app.flags.DEFINE_string("ckpt_dir", "../ckpt", "check point directory.")
FLAGS = tf.app.flags.FLAGS
########################################
class Robot:
	def __init__(self):
		self.du = DU()
		self.vocab ,self.recab = self.du.initialize_vocabulary()
		self.ids_arr= []
		for line in open(self.du.ids_path):
			line = line.strip()
			if len(line) > 0:
				temp  =line.split(' ')
				for i in range(len(temp)):
					temp[i] = int(temp[i])
				self.ids_arr.append(temp)
			else:
				self.ids_arr.append([])
		
		self.mark = json.load(open(self.du.mark_path))
		self.train = json.load(open(self.du.train_path))
		self.dev = json.load(open(self.du.dev_path))
		self.test = json.load(open(self.du.test_path))
			
		self.model = Ranker(
				vocab_size     = FLAGS.vocab_size,
				embedding_size = FLAGS.emd_size,
				memory_size    = FLAGS.mem_size,
				batch_size     = FLAGS.batch_size,
				max_dialogue_size = FLAGS.max_dialogue_size,
				max_sentence_size = FLAGS.max_sentence_size,
				margin         = FLAGS.margin,
				max_gradient_norm = FLAGS.max_gradient_norm,
				learning_rate  = FLAGS.learning_rate,
				learning_rate_decay_factor = FLAGS.learning_rate_decay_factor,
				use_lstm       = False,
				train_mode     = FLAGS.train
				)
	def build_model(self,session):
		self.model.build_model()
		ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
		#尝试从检查点恢复模型参数
		if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
			print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
			self.model.saver.restore(session, ckpt.model_checkpoint_path)
		else:
			print("Created model with fresh parameters.")
			session.run(tf.initialize_all_variables())

	def ut2ids(self,ut):	#将句子标记转换为具体的词id列表
		#返回单个对话机器candidates的id表示
		if ut ==None or len(ut)==0:
			return None
		result =  []
		for i in range(len(ut)):
			cache = []
			for j in range(len(ut[i])):
				cache.append(self.ids_arr[ut[i][j]])
			result.append(cache)
		return result
	
	def run_train(self):
		train_set = json.load(open(self.du.train_path,'r'))
		if len(train_set) > FLAGS.max_trainset_size and FLAGS.max_trainset_size!=0:
			train_set = train_set[:FLAGS.max_trainset_size]
		dev_set = json.load(open(self.du.dev_path,'r'))
		if len(dev_set) > FLAGS.max_devset_size and FLAGS.max_devset_size!=0:
			dev_set = dev_set[:FLAGS.max_devset_size]
		
		with tf.Session() as sess:
			self.build_model(sess)
			step_time, loss = 0.0, 0.0
			current_step = 0
			previous_losses = []
			step_count = 0
			while True:
				# Get a batch and make a step.
				start_time = time.time()
				dialogs = []
				for i in range(FLAGS.batch_size):
					temp = random.choice(train_set)
					for i in range(len(temp)):	#遍历选出对话的每句
						if i%2==0:
							continue
						true_candidate = temp[i][0]	#正确答案
						seed = int(random.uniform(0,len(self.mark)-1))	#候选的 错误答案
						while self.mark[seed]==False or cmp(true_candidate,self.ids_arr[seed])==0:
							# 如果依据话是客户说的或者和正确内容重复,则重新选取
							seed = int(random.uniform(0,len(self.mark)-1))
						temp[i][1] = seed	#对错误答案进行赋值
					dialogs.append(self.ut2ids(temp))
				history_batch,true_batch,false_batch = self.model.train2vec(dialogs,step_count)
				#获取id的表达
				step_loss = self.model.step_train(sess,history_batch,true_batch,false_batch)
				#进行一步训练
				step_count += 1
				step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
				loss += step_loss / FLAGS.steps_per_checkpoint
				#print("step_count",step_count,"step_loss",step_loss)
					
				if step_count % FLAGS.steps_per_checkpoint == 0:  #统计数据，保存模型
										#统计在训练过程中时间和平均损失等信息
					pat1, pat3, MAP, count = 0.0, 0.0, 0.0, 0.0
					for i in range(len(dev_set)):
						dialog = self.ut2ids(dev_set[i])
						dialog,candidates  = self.model.test2vec(dialog)
						scores = self.model.step_test(sess,dialog,candidates)
						scores = scores[:int(len(dev_set[i])/2)]
						a, b, c, d = self.cal_score(scores)
						pat1 += a
						pat3 += b
						MAP += c
						count += d
					length = count
					print('Validation Set Result: P@1 %.4f, P@3 %.4f MAP %.4f'%(pat1/length,pat3/length,MAP/length))

					print ("global step %d learning rate %.4f step-time %.2f average loss %.2f" 
							%(self.model.global_step.eval(), self.model.learning_rate.eval(), step_time, loss))
					loss = MAP/count
					if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
						sess.run(self.model.learning_rate_decay_op)
					previous_losses.append( loss)
					checkpoint_path = os.path.join(FLAGS.ckpt_dir, "hred.ckpt")
					self.model.saver.save(sess, checkpoint_path, global_step = self.model.global_step)
					step_time, loss = 0.0, 0.0

						
	def cal_score(self,scores):
		pat1=  0.0
		pat3 = 0.0
		MAP = 0.0
		length = len(scores)
		for i in range(length):
			rank = 1.0
			for j in range(len(scores[i])):
				if scores[i][0] < scores[i][j]:
					rank += 1
			if rank == 1:
				pat1 += 1

			if rank <= 3:
				pat3 += 1

			MAP += 1/rank

		return pat1, pat3, MAP, length
			
					
					
	def run_test(self):
		self.build_model()
		test_set = json.load(open(self.du.test_path,'r'))
		pat1, pat3, MAP = 0.0, 0.0, 0.0
		for i in range(len(test_set)):
			dialog = self.ut2ids(test_set[i])
			dialog,candidates  = self.model.test2vec(dialog)
			scores = self.model.step_test(sess,dialog,candidates)
			scores = scores[:int(len(dev_set[i]/2))]
			a, b, c = self.cal_score(scores)
			pat1+=a
			pat3+=b
			MAP+=c
		length = len(test_set)
		print('Test Set Result: P@1 %.4f, P@3 %.4f MAP %.4f'%(pat1/length,pat3/length,MAP/length))

def main(_):
	robot = Robot()
	if FLAGS.train:
		robot.run_train()
	else:
		robot.run_test()

if __name__ == '__main__':
	tf.app.run()
