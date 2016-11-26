#coding=utf-8
import csv
class Robot:
	def __init__(self):
		self.in_path = 'skin.csv'
		self.out_path = 'skin.txt'
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
			for row in spamreader:
				if first:
					first = False
					continue
				current_id = row[2]
				content = row[6]
				content = content.replace("\n","").replace("^","").replace("\t","")
				role = row[4]
				
				session_type= row[1]
				if current_id !=cache_id:
					role_cache = role+"a"
					if len(context_cache)>25 or len(context_cache)<5:
						context_cache =[]
					else:
						out.write("^".join(context_cache)+"\n")
						context_cache =[]
				cache_id = current_id
#				print(role)
#				print(type(role))
#print(type("卡片"))
				p1 = unicode("卡片","utf-8")
				p2= unicode(row[3],"utf-8")
#			print(", ".join(row))
#			print(p1,p2)
				if p1 in p2:
#					print(", ".join(row))
					continue
					
				if role_cache!=role:
					context_cache.append(content)
				else:
					context_cache[-1]+="。"+content
				role_cache = role
			print (",".join(row))
			print (row[6])
				
				
			#print ', '.join(row)
if __name__ == '__main__':
	temp = Robot()
	temp.run()
