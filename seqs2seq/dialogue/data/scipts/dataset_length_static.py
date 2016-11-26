#coding=utf-8
import csv
class Robot:
##统计结果 80%的session交互次数小于25
##80% 的句子长度小于59个词 (实际上是20个中文单词)
	def __init__(self):
		self.in_path = 'skin.csv'
		self.out_path = 'skin.txt'
	def run(self):
		length_static ={}
		sent_len_count = {}
		with open(self.in_path,'rb') as csvfile:
			#spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			spamreader = csv.reader(csvfile)
			current_length = 0
			first = True
			cache_id = ""
			current_id = ""
			row_counter =0
			sentence_counter=0
			for row in spamreader:
				if first:
					first = False
					continue
				current_id = row[2]
				content = row[6]
				content_len = len(content)
				sent_len_count[content_len] = sent_len_count.get(content_len,0)+1
				sentence_counter+=1
				if current_id !=cache_id:
					length_static[current_length] = length_static.get(current_length,0)+1
					row_counter+=1
					current_length =0
				cache_id = current_id
				current_length+=1
			print (",".join(row))
			print (row[6])
			#print(length_static)
			sum=0
			
			print("各个session中对话数的分布")
			for key in length_static:
				sum+=length_static[key]
				print(key,sum,sum*1.0/row_counter)

			sum =0
			print("每条语句中的字符数量")
			for key in sent_len_count:
				sum+=sent_len_count[key]
				print(key,sum*1.0/sentence_counter)
				
				
			#print ', '.join(row)
if __name__ == '__main__':
	temp = Robot()
	temp.run()
