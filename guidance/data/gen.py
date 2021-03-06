#coding=utf-8
import json
class Gen:
	def __init__(self):
		self.data_path = 'tag_temp.data'
		self.tag_path  = 'tag.data'
		self.ut_path = 'ut.data'
		self.train_path = 'train.data'
		self.dev_path = 'dev.data'
		self.test_path = 'test.data'
		self.all_data = []
	def get_tags(self):
		tag_set={}
		for line in open(self.data_path,'r'):
			line =line.strip()
			items =line.split('\t')
			if len(items)==3:
				tags = items[2].split('/')
				for tag in tags:
					tag_set[tag] = tag_set.get(tag,0)+1
			else:
				pass
				#print(line)
		tag_file = open(self.tag_path,'w')
		tuples =  sorted(tag_set.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
		print('tag_size',len(tuples))
		for i in range( min(5000,len(tuples))):
			tag_file.write(tuples[i][0]+'\n')

	def build_all(self):
		tags = {}
		index =0
		for line in open(self.tag_path,'r'):
			line =line.strip()
			tags[line] = index
			index +=1
		ut_file = open(self.ut_path,'w')
		ut_index = 0
		for line in open(self.data_path,'r'):
			line =line.strip()
			items =line.split('\t')
			if len(items)==3:
				match = []
				for i in items[2].split('/'):
					i=i.decode('utf-8')
					if tags.has_key(i):
						match.append(tags[i])
				if len(match)!=0:
					title = items[0].strip().replace('\n','').replace('\t','').replace('  ',' ')
					body = items[1].strip().replace('\n','').replace('\t','').replace('  ',' ')
					if len(title)+len(body)<500:
						ut_file.write(title+'#'+body+'\n')
						
						self.all_data.append([ut_index,match])
						ut_index+=1

	def split(self):
		train_file = open(self.train_path,'w')
		dev_file = open(self.dev_path,'w')
		test_file = open(self.test_path,'w')
		size = len(self.all_data)
		json.dump(self.all_data[:int(size*0.7)],train_file)
		json.dump(self.all_data[int(size*0.7):int(size*0.8)],dev_file)
		json.dump(self.all_data[int(size*0.8):],test_file)

		
		
if __name__ =='__main__':
	temp = Gen()
	temp.get_tags()
	temp.build_all()
	temp.split()
			
			
