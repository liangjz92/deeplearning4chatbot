if __name__ == '__main__':
	full_path = 'skin.data'
	count = 0
	for line in open(full_path,'r'):
		count +=1
	train_mark = int(count*0.7)
	dev_mark = int(count*0.8)

	train_path = 'skin-train.data'
	dev_path  = 'skin-dev.data'
	test_path  = 'skin-test.data'
	out_file = open(train_path,'w')
	count = 0
	for line in open(full_path,'r'):
		if count == train_mark:
			out_file.close()
			out_file = open(dev_path,'w')
		if count == dev_mark:
			out_file.close()
			out_file = open(test_path,'w')	
		out_file.write(line)
		count = count+1
	out_file.close()

