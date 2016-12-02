#coding=utf-8
import jieba
import re
import sys
import os
import random
import json
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

class Maker:
	def __init__(self,size):
		self.data_path = 'skin.data'
		self.train_size = int(size*0.7)
		self.dev_size = int(size*0.1)
		self.test_size = size - self.train_size - self.dev_size
		jieba.load_userdict('medical.txt')
		self.sentences = []
		self.orders = []
		self.stop_line = []
		for line in open('goodbye.data'):
			line = line.strip()
			self.stop_line.append(line)
		self.ac_dialogs = []
	def step_1(self):
		cache = {}
		for line in open(self.data_path):
			temp = []
			sess =line.strip().split('^')
			if len(sess)<10 or len(sess)> 25:
				continue
			for i in range(len(sess)):
				if i%2==1:
					#客服说的话
					if ('我是' in sess[i]) and ('师' in sess[i]):
						sess[i] = '您好，我是药师，请您做下症状描述'
					stop_mark = False
					for item in self.stop_line:
						if item in sess[i]:
							stop_mark = True
							break
					if stop_mark:
						i = len(sess)
						continue
					#print('host:'+sess[i].decode('utf-8'))
					cache[sess[i]] = cache.get(sess[i],0)+1
				temp.append(sess[i])
				#else:
					#print('user:'+sess[i].decode('utf-8'))
			if not '用药咨询' in temp[0]:
				self.ac_dialogs.append(temp)	
		x = sorted(cache.items(), lambda x, y: cmp(y[1], x[1]))
		for item in x[100:500]:
			if len(item[0])>30:
				pass
				#print(item[0])
			#print(item[1])
	
	def step_2(self):
		pimg = re.compile(r'<img src=\'.*\'/>')
		pitem = re.compile(r'<a target=.*>查看大图</a>')
		for sess in self.ac_dialogs:
			for i in range(len(sess)):
				if '<' in sess[i]:
					#print(sess[i])
					sess[i] = re.sub(pimg,'图片链接',sess[i])
					sess[i] = re.sub(pitem,'产品链接',sess[i])
	def step_3(self):
		for i in range(len( self.ac_dialogs)):
			sess = self.ac_dialogs[i]
			if '请您做下症状描述' in sess[1]:
				
#print(sess[1])
				if sess[0]!=sess[2]:
					sess[2] = sess[0] +'。'+sess[2]
				sess= sess[2:]
				self.ac_dialogs[i] = sess
#				print(sess[1])
		
	def step_4(self):
		self.ut_arr = []
		self.ut_mark= []
		self.id_arr = []
		for sess in self.ac_dialogs:
			temp = []
			for i  in range(len(sess)):
				ut = sess[i]
				self.ut_arr.append(ut)
				if i%2==0:
					self.ut_mark.append(False)
				else:
					self.ut_mark.append(True)
				temp.append([len(self.ut_arr)-1])
			self.id_arr.append(temp)
	def step_5(self):
		for sess in self.id_arr:
			for i in range(len(sess)):
				if i%2==0:
					continue
				cache = set()
				cache.add(self.ut_arr[sess[i][0]])
				while(len(cache)< 100):
					seed = int(random.uniform(0,len(self.ut_arr)-1))
					if self.ut_mark[seed]==False:	#客户说的话
						continue
					if self.ut_arr[seed] in cache:
						continue
					cache.add(self.ut_arr[seed])
					sess[i].append(seed)
				#for i in sess[i]:
				#	print(self.ut_arr[i])
				#print('#############')
		#print(sess)
	def step_6(self):
		ut_file = open('uterance.data','w')
		mark_file = open('mark.data','w')
		train = open('train.data','w')
		dev = open('dev.data','w')
		test = open('test.data','w')
		for line in self.ut_arr:
			ut_file.write(line + '\n')
		json.dump(self.ut_mark,mark_file)
		all_count = len(self.id_arr)
		json.dump(self.id_arr[:int(all_count*0.7)],train)
		json.dump(self.id_arr[int(all_count*0.7):int(all_count*0.8)],dev)
		json.dump(self.id_arr[int(all_count*0.8):],test)
	

	def show(self):
		for sess in self.ac_dialogs:
			print('################')
			for i in range(len(sess)):
				if i%2==0:
					print('user:'+sess[i].decode('utf-8'))		
				else:
					print('host:'+sess[i].decode('utf-8'))
	
	def dump(self):
		for sess in self.ac_dialogs:
			print('^'.join(sess))


if __name__ == '__main__':
	size = 500000
	if len(sys.argv) >=2:
		size = int(sys.argv[1])
	temp = Maker(size)
	temp.step_1()
	temp.step_2()
	temp.step_3()
	temp.step_4()
	temp.step_5()
	temp.step_6()
	#temp.show()
	#temp.dump()

		
