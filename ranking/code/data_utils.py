#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import jieba
from six.moves import urllib
from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
SPLITER = "^"

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


class DU:
	def __init__(self):
		self.ut_path = '../data/uterance.data'
		self.mark_path = '../data/mark.data'
		self.vocab_path = '../data/vocab.data'
		self.ids_path = '../data/ids.data'
		self.train_path = '../data/train.data'
		self.dev_path = '../data/dev.data'
		self.test_path = '../data/test.data'
		self.dict_path = '../data/medical.txt'
		self.emd_path = '../data/emd/ylemd.bin'
		jieba.load_userdict(self.dict_path)
	
	def basic_tokenizer(self,sentence):
		words = []
		for space_separated_fragment in sentence.strip().split():
			words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
		return [w for w in words if w]

	def jieba_tokenizer(self,sentence):
		return jieba.lcut(sentence)

	def create_vocabulary(self, max_vocabulary_size,tokenizer=None, normalize_digits=False):
		vocabulary_path = self.vocab_path
		data_path = self.ut_path
		print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
		vocab = {}

		with gfile.GFile(data_path, mode="rb") as f:
			counter = 0
			for line in f:
				counter += 1
				if counter % 100000 == 0:
					print("	processing line %d" % counter)
				tokens = self.jieba_tokenizer(line)
				for w in tokens:
					word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
					if len(word)>10:
						continue		
					if word in vocab:
						vocab[word] += 1
					else:
						vocab[word] = 1
			vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
			print('full vacab size:',len(vocab_list))
			if len(vocab_list) > max_vocabulary_size:
				vocab_list = vocab_list[:max_vocabulary_size]
			with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
				for w in vocab_list:
					#print (w)
					vocab_file.write((w + b"\n").encode("utf-8"))

	def initialize_vocabulary(self):
		vocabulary_path = self.vocab_path
		if gfile.Exists(vocabulary_path):
			rev_vocab = []
			with gfile.GFile(vocabulary_path, mode="rb") as f:
				rev_vocab.extend(f.readlines())
			rev_vocab = [line.strip() for line in rev_vocab]
			vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
			return vocab, rev_vocab
		else:
			raise ValueError("Vocabulary file %s not found.", vocabulary_path)

	def sentence_to_token_ids(self, sentence, vocabulary,tokenizer=None, normalize_digits=True):
		words = self.jieba_tokenizer(sentence)
		words = [w.encode('utf-8') for w in words]
		if not normalize_digits:
			return [vocabulary.get(w, UNK_ID) for w in words]
		# Normalize digits by 0 before looking words up in the vocabulary.
		return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]

	def data_to_token_ids(self,  tokenizer=None, normalize_digits=True):
		data_path  =self.ut_path
		target_path = self.ids_path
		vacab_path = self.vocab_path
		print("Tokenizing data in %s" % data_path)
		vocab, _ = self.initialize_vocabulary()
		with gfile.GFile(data_path, mode="rb") as data_file:
			with gfile.GFile(target_path, mode="w") as tokens_file:
				counter = 0
				for line in data_file:
					line = line.strip()
					sentence_array = line.split(SPLITER)
					token_array= []
					for sentence in sentence_array:
						token_ids = self.sentence_to_token_ids(sentence, vocab, tokenizer,normalize_digits)
						token_array.append(" ".join([str(tok) for tok in token_ids]))
					counter += 1
					if counter % 100000 == 0:
						print("	tokenizing line %d" % counter)
					tokens_file.write("\t".join(token_array) + "\n")

if __name__ =="__main__":
	temp = DU()
	temp.create_vocabulary(30000,False)
	temp.data_to_token_ids()
	
	
