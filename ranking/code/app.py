#coding = utf-8
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
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 100.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("margin", 0.05, "margin between true and false candiate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("emd_size", 300, "embedding size")
tf.app.flags.DEFINE_integer("mem_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size", 30000, "vocabulary size.")
tf.app.flags.DEFINE_string("ckpt_dir", "../ckpt", "check point directory.")
tf.app.flags.DEFINE_integer("max_dialogue_size", "25", "how manay uts in one sess max")
tf.app.flags.DEFINE_integer("max_sentence_size", "36", "how manay tokens in one sentence max")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
		                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer('max_dev_data_size',0,"how many dev samples use max")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
		                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("train", True,
		                            "True to train model, False to decode model")
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
		self.model.build_model()
	
	def run_train(self):
		pass
	
	def run_test(self):
		pass

def main(_):
	robot = Robot()

if __name__ == '__main__':
	tf.app.run()
