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
import data_utils
from data_utils import DU
from marker import Marker
import json
########################################
tf.app.flags.DEFINE_float("learning_rate", 0.25, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.5, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("emd_size", 300, "embedding size")
tf.app.flags.DEFINE_integer("mem_size", 64, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size", 40001, "vocabulary size.")
tf.app.flags.DEFINE_integer("tag_size", 180, "tag size.")
tf.app.flags.DEFINE_integer("max_ut_size", "20", "how manay tokens in one sentence max 36")
tf.app.flags.DEFINE_integer("max_trainset_size", 1000000, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer('max_devset_size',100,"how many dev samples use max")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000, "aHow many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("train", True, "True to train model, False to decode model")
tf.app.flags.DEFINE_string("ckpt_dir", "../ckpt", "check point directory.")
FLAGS = tf.app.flags.FLAGS
########################################
class Robot:
	def __init__(self):
		self.du = DU()
		self.vocab ,self.recab = self.du.initialize_vocabulary()
		self.tag,self.retag = self.du.init_tag()	#载入标签和对应的id
		self.ids_arr= []
		for line in open(self.du.ids_path):
			line = line.strip()
			if len(line) > 0:
				temp  =line.split(' ')
				for i in range(len(temp)):
					try:
						temp[i] = int(temp[i])
					except Exception:
						temp[i] = 3
				self.ids_arr.append(temp)
			else:
				self.ids_arr.append([])
		
#		self.train = json.load(open(self.du.train_path))
#		self.dev = json.load(open(self.du.dev_path))
#		self.test = json.load(open(self.du.test_path))
			
		self.model = Marker(
				vocab_size     = FLAGS.vocab_size,
				embedding_size = FLAGS.emd_size,
				memory_size    = FLAGS.mem_size,
				label_size     = FLAGS.tag_size,
				batch_size     = FLAGS.batch_size,
				max_ut_size    = FLAGS.max_ut_size,
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
			emd_weight = np.random.rand(FLAGS.vocab_size,FLAGS.emd_size)*0.16-0.08
			f = open(self.du.emd_path,'r')
			first=True
			for line in f:
				if first:
					first=False
					continue
				box = line.split(' ')
				word = box[0]
				box = box[1:FLAGS.emd_size+1]
				if self.vocab.has_key(word):
					index = self.vocab[word]
					one_emd = np.array([float(x) for x in box])
					emd_weight[index,:] = one_emd
			load = self.model.embedding_weight.assign(emd_weight)
			session.run(load)
			print('word embedding load over')
		self.train_writer = tf.train.SummaryWriter('../summary',session.graph)

	def ut2ids(self,ut):	#将句子标记转换为具体的词id列表
		#返回单个对话机器candidates的id表示,包含batch
		#print(ut)
		if ut ==None or len(ut)==0:
			return None
		result =[]
		for i in range(len(ut)):
			temp =[]
			temp.append(self.ids_arr[ut[i][0]])
			temp.append(ut[i][1])
			result.append(temp)
		return result

	def ids2ut(self,ids):	#将id转换为文本单词句子
		if ids ==None or len(ids) ==0:
			return None
		ut = []
		for i in range(len(ids)):
			ut.append(self.recab[ids[i]])
		return ' '.join(ut)
	
	def run_train(self):
		print('Running train op')
		train_set = json.load(open(self.du.train_path,'r'))
		train_set2 = train_set[:100]	#用于测试模型在训练集上的得分
		if len(train_set) > FLAGS.max_trainset_size and FLAGS.max_trainset_size!=0:
			train_set = train_set[:FLAGS.max_trainset_size]
		dev_set = json.load(open(self.du.dev_path,'r'))
		if len(dev_set) > FLAGS.max_devset_size and FLAGS.max_devset_size!=0:
			dev_set = dev_set[:FLAGS.max_devset_size]
#		with tf.device('/cpu:0'):
#			sess = tf.Session()
		with tf.Session() as sess:
			self.build_model(sess)
			step_time, loss = 0.0, 0.0
			step_count = self.model.global_step.eval()
			previous_losses = []
			while True:
				start_time = time.time()
				samples =[]
				for i in range(FLAGS.batch_size):
					temp = random.choice(train_set)	#随机挑选一个样本
					samples.append(temp)
				sample = self.ut2ids(samples)
				ut_arr,labels = self.model.sample2vec(sample)
				step_loss,summary = self.model.step_train(sess,ut_arr,labels)
				self.train_writer.add_summary(summary,step_count)
				
				step_count += 1
				step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
				loss += step_loss / FLAGS.steps_per_checkpoint
					
				if step_count % FLAGS.steps_per_checkpoint == 0:  #统计数据，保存模型
					for data_set in [dev_set,train_set2]:
						pat1, pat3, MAP, count = 0.0, 0.0, 0.0, 0.0
						samples =[]
						labels_cache=[]
						for i in range(len(data_set)):
							samples.append(data_set[i])
							labels_cache.append(data_set[i][1])
						samples = self.ut2ids(samples)
						ut_arr,labels = self.model.sample2vec(samples)
						scores = self.model.step_test(sess,ut_arr,labels)
						hit_count,all_count = self.cal_score(scores,labels_cache)
						print('top2 hit:\t%.4f'%(hit_count/all_count))
					print ("global step %d learning rate %.4f step-time %.4f average loss %.4f" 
							%(self.model.global_step.eval(), self.model.learning_rate.eval(), step_time, loss))
					if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
						sess.run(self.model.learning_rate_decay_op)
					previous_losses.append( loss)
					checkpoint_path = os.path.join(FLAGS.ckpt_dir, "hred.ckpt")
					self.model.saver.save(sess, checkpoint_path, global_step = self.model.global_step)
					step_time, loss = 0.0, 0.0

						
	def cal_score(self,scores,labels):
		#计算top2的平均准确率
		#print(scores.shape)
		#print(labels)
		batch_size = scores.shape[0]
		hit_count = 0.0
		all_count = 0.0
		for i in range(batch_size):
			logit = scores[i,:]
			tops = logit.argsort()[-2:][::-1]
			#print('logit',tops)
			#print('labels',labels[i])
			tags =set(labels[i])
			for j in range(len(tops)):
				#print(tops[j],logit[tops[j]])
				if tops[j] in tags:
					hit_count = hit_count+1
			#for k in tags:
				#print(k,logit[k])
			all_count = all_count + len(tags)
		return hit_count,all_count
			
					
					
	def run_test(self):
		print('Running Test Op')
		#模型测试
		with tf.Session() as sess:
			self.build_model(sess)
			test_set = json.load(open(self.du.test_path,'r'))
			#test_set = json.load(open(self.du.train_path,'r'))
			cache = []
			label_cache =[]
			hit_count,all_count = 0.0,0.0
			print('length of test_set:',len(test_set))
			for i in range(len(test_set)):
				cache.append(test_set[i])
				label_cache.append(test_set[i][1])
				if i%200==0 or i==(len(test_set)-1):
					samples = self.ut2ids(cache)
					ut_arr,labels = self.model.sample2vec(samples)
					scores = self.model.step_test(sess,ut_arr,labels)
					h,a = self.cal_score(scores,label_cache)
					hit_count +=h
					all_count +=a
					cache,label_cache = [],[]
			print('Test Set Top2 Score:\t%.4f'%(hit_count/all_count))

	
def main(_):
	robot = Robot()
	if FLAGS.train:
		robot.run_train()
	else:
		robot.run_test()

if __name__ == '__main__':
	tf.app.run()
