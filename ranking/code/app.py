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
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 15.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("margin", 0.3, "margin between true and false candiate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("emd_size", 300, "embedding size")
tf.app.flags.DEFINE_integer("mem_size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size", 30001, "vocabulary size.")
tf.app.flags.DEFINE_integer("max_dialogue_size", "10", "how manay uts in one sess max,25")
tf.app.flags.DEFINE_integer("max_sentence_size", "20", "how manay tokens in one sentence max 36")
tf.app.flags.DEFINE_integer("max_trainset_size", 1000000, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer('max_devset_size',100,"how many dev samples use max")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_float("drop_out", 1.0, "keep prob")
tf.app.flags.DEFINE_integer("layer", 1, "rnn layer")
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
				train_mode     = FLAGS.train,
#				drop_out 	   = FLAGS.drop_out,
#				layer		   = FLAGS.layer
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
			'''session.run( 
				tf.initialize_variables( 
					list( 
						tf.get_variable(name) for name in session.run( 
							tf.report_uninitialized_variables( 
								tf.all_variables() 
			) ) ) ) )'''
			session.run(tf.initialize_all_variables())
			
			#emd_weight = tf.random_normal([FLAGS.vocab_size,FLAGS.emd_size],-0.08,0.08)
			emd_weight = np.random.rand(FLAGS.vocab_size,FLAGS.emd_size)*0.16-0.08
			#f = open('../data/emd/ylemd-128.bin','r')
			f = open('../data/emd/ylemd.bin','r')
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

	def ids2ut(self,ids):	#将id转换为文本单词句子
		if ids ==None or len(ids) ==0:
			return None
		ut = []
		for i in range(len(ids)):
			ut.append(self.recab[ids[i]])
		return ' '.join(ut)
	
	def run_train(self):
		print('running train op')
		train_set = json.load(open(self.du.train_path,'r'))
		train_set2 = train_set[:100]
		if len(train_set) > FLAGS.max_trainset_size and FLAGS.max_trainset_size!=0:
			train_set = train_set[:FLAGS.max_trainset_size]
		
		dev_set = json.load(open(self.du.dev_path,'r'))
		if len(dev_set) > FLAGS.max_devset_size and FLAGS.max_devset_size!=0:
			dev_set = dev_set[:FLAGS.max_devset_size]
		
		with tf.Session() as sess:
			self.build_model(sess)
			step_time, loss = 0.0, 0.0
			step_count = self.model.global_step.eval()
			previous_losses = []
			print('P@1\tP@3\tMAP\tStep\tLR\tTime\tLoss')
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

				step_loss,summary = self.model.step_train(sess,history_batch,true_batch,false_batch)
				if step_count%5==0:
					self.train_writer.add_summary(summary,step_count)
				
				#进行一步训练
				step_count += 1
				step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
				loss += step_loss / FLAGS.steps_per_checkpoint
					
				if step_count % FLAGS.steps_per_checkpoint == 0:  #统计数据，保存模型
										#统计在训练过程中时间和平均损失等信息
					pat1, pat3, MAP, count = 0.0, 0.0, 0.0, 0.0
					for i in range(len(dev_set)):
						dialog = self.ut2ids(dev_set[i])
						dialog1 = dialog[:]
						dialog,candidates  = self.model.test2vec(dialog)
						scores = self.model.step_test(sess,dialog,candidates)
						'''
						for j in range(min(len(scores)*2,len(dialog1))):
							if j%2==0:
								print('user',self.ids2ut(dialog1[j][0]))
							else:
								#print(dialog1[i])
								print('host',scores[int(j/2)][0],self.ids2ut(dialog1[j][0]))
								max_index = np.argmax(scores[int(j/2)])
								print('max_cand',max_index,scores[int(j/2)][max_index],self.ids2ut(dialog1[j][max_index]))
								print('')
						print('############新对话################')	
						'''
						scores = scores[:int(len(dev_set[i])/2)]
						a, b, c, d = self.cal_score(scores)
						pat1 += a
						pat3 += b
						MAP += c
						count += d
					length = count
					#print('P@1\tP@3\tMAP\tStep\tLR\tTime\tLoss')
					print('V100: %.4f\t%.4f\t%.4f\t%.0f\t%.4f\t%.4f\t%.4f'
						%(pat1/length,pat3/length,MAP/length,self.model.global_step.eval(), self.model.learning_rate.eval(), step_time, loss))
					pat1, pat3, MAP, count = 0.0, 0.0, 0.0, 0.0
					for i in range(len(train_set2)):
						dialog = self.ut2ids(train_set2[i])
						dialog1 = dialog[:]
						dialog,candidates  = self.model.test2vec(dialog)
						scores = self.model.step_test(sess,dialog,candidates)
						scores = scores[:int(len(train_set2[i])/2)]
						a, b, c, d = self.cal_score(scores)
						pat1 += a
						pat3 += b
						MAP += c
						count += d
					length = count
					print('T100: %.4f\t%.4f\t%.4f\t%.0f\t%.4f\t%.4f\t%.4f'
						%(pat1/length,pat3/length,MAP/length,self.model.global_step.eval(), self.model.learning_rate.eval(), step_time, loss))
					#print('Validation Set Result: P@1 %.4f, P@3 %.4f MAP %.4f'%(pat1/length,pat3/length,MAP/length))

					#print ("global step %d learning rate %.4f step-time %.4f average loss %.4f" 
					#		%(self.model.global_step.eval(), self.model.learning_rate.eval(), step_time, loss))
					loss = MAP/count
					if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
						sess.run(self.model.learning_rate_decay_op)
					previous_losses.append( loss)
					if len(previous_losses)%4==0:	#三次测试保存一次模型
						checkpoint_path = os.path.join(FLAGS.ckpt_dir, "hred.ckpt")
						self.model.saver.save(sess, checkpoint_path, global_step = self.model.global_step)
					step_time, loss = 0.0, 0.0

						
	def cal_score(self,scores):
		pat1=  0.0
		pat3 = 0.0
		MAP = 0.0
		length = len(scores)
		for i in range(length):
			#print('scores:',i,scores[i])
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
		print('Running Test Op')
		#模型测试
		with tf.Session() as sess:
			self.build_model(sess)
			#test_set = json.load(open(self.du.test_path,'r'))
			test_set = json.load(open(self.du.train_path,'r'))
			pat1, pat3, MAP,count = 0.0, 0.0, 0.0, 0.0
			for i in range(len(test_set)):
				dialog = self.ut2ids(test_set[i])
				dialog1 = dialog[:]
				dialog,candidates  = self.model.test2vec(dialog)
				scores = self.model.step_test(sess,dialog,candidates)
				'''
				for j in range(min(len(scores)*2,len(dialog1))):
					if j%2==0:
						print('user',self.ids2ut(dialog1[j][0]))
					else:
						#print(dialog1[i])
						print('host',scores[int(j/2)][0],self.ids2ut(dialog1[j][0]))
						max_index = np.argmax(scores[int(j/2)])
						print('max_cand',max_index,scores[int(j/2)][max_index],self.ids2ut(dialog1[j][max_index]))
						print('')
				print('############新对话################')	
				'''
				scores = scores[:int(len(test_set[i])/2)]
				a, b, c ,d= self.cal_score(scores)
				pat1+=a
				pat3+=b
				MAP+=c
				count +=d
			length = count
			print('Test Set Result: P@1 %.4f, P@3 %.4f MAP %.4f'%(pat1/length,pat3/length,MAP/length))

def main(_):
	robot = Robot()
	if FLAGS.train:
		robot.run_train()
	else:
		robot.run_test()

if __name__ == '__main__':
	tf.app.run()
