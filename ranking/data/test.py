import json
data = open('test.data','r')
data =json.load(data)
for line in data:
	if len(line)  < 3:
		print(line)
