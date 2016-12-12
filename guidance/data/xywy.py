#coding=utf-8
import csv
inf = csv.reader(file('xywy.csv','rb'))
outf = open('xywy.data','w')
for line in inf:
	box =line
	tag1 = box[1]
	tag2 =box[2]
	msg = box[6].replace('\t','').replace('\n','')
	out = msg+'\t'+tag1+'\t'+tag2+'\n'
	outf.write(out)

	 
	

