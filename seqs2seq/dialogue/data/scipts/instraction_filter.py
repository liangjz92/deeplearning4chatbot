#coding=utf-8
import csv
class Robot:
	def __init__(self):
		self.in_path = 'skin.csv'
		self.out_path = 'instrucation.txt'
	def run(self):
		out = open(self.out_path,'w')
		with open(self.in_path,'rb') as csvfile:
			#spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			spamreader = csv.reader(csvfile)
			first = True
			cache_id = ""
			current_id = ""
			context_cache = []
			role_cache =""
			txt = ""
			max_length = 0
			for row in spamreader:
				if first:
					first = False
					continue
				current_id = row[2]
				content = row[6]
				content = content
				role = row[4]
				
				session_type= row[1]
				if current_id !=cache_id:
					role_cache = role+"a"
					if len(context_cache)>25 or len(context_cache)<5 or max_length<360:
						context_cache =[]
					else:
						for line in context_cache:
							out.write(line+"\n")
						context_cache =[]
					max_length =0
				cache_id = current_id
				context_cache.append(",".join(row))
				max_length = max(max_length,len(content))
				
				
if __name__ == '__main__':
	temp = Robot()
	temp.run()
