import random
import numpy as np
random.seed(5)
class Dataset():

	'Characterizes a dataset'
	def __init__(self, ID, data_label, data_path):
		'Init'
		self.bucket = []
		self.bucket_limit = 1
		self.line_counter = 0
		self.data_label = data_label
		self.ID = ID
		self.path = data_path
		self.eof = False
		self.multiplier = 1

	def __len__(self):
		'number of samples'
		return len(self.ID)

	def __getitem__(self):
		'Generates one sample of data'
		# Select sample
		if len(self.bucket) <= 0 and self.eof == True:
			print("out of data")
			return -1,-1

		if len(self.bucket) <= 0:
			ret = self.fill_bucket()
			if ret == -1:
				print("Read in all data")
				self.eof = True

		temp = self.bucket.pop()
		X_input = temp[0:-1]
		X_input = [float(i) for i in X_input]
		y_output = float(temp[-1].replace(',','.')) * self.multiplier
		return (X_input, y_output)

	def reset(self):
		self.bucket = []
		self.line_counter = 0

	def fill_bucket(self):
		temp_buffer = []
		f_in = open(self.path,'r')
		i = self.line_counter
		end = i + 1000
		begin = 0
		while(begin < i):
			f_in.readline()
			begin+=1
		while(i < end):
			line = f_in.readline()
			if line == None or len(line) < 10:
				return -1
			line = line[:-1]
			split = line.split(" ")

			split = (list)(filter(lambda a: a != '', split))
			if len(split) != 12:
				a = split[6]
				a = a.split('-')
				del split[6]
				split.insert(6,a[0])
				split.insert(7,"-"+a[1])

			self.bucket.append(split)
			i+=1
			self.line_counter += 1
		f_in.close()
		return 0

#testing stuff
class ConvDataset():

	'Characterizes a dataset'
	def __init__(self, ID, data_label, data_path):
		'Init'
		self.bucket = []
		self.bucket_limit = 1
		self.item_count = 0
		self.data_label = data_label
		self.ID = ID
		self.path = data_path
		self.eof = False
		self.multiplier = 1

	def __len__(self):
		'number of samples'
		return len(self.ID)

	def __getitem__(self):
		'Generates one sample of data'
		# Select sample
		if len(self.bucket) <= 0 and self.eof == True:
			print("out of data")
			return -1,-1

		if len(self.bucket) <= 0:
			ret = self.fill_bucket()
			if ret == -1:
				print("Read in all data")
				self.eof = True

		r = random.SystemRandom()
		r.shuffle(self.bucket)
		temp = self.bucket.pop()
		X_input = temp[0:-1]
		y_output = temp[-1]
		return (X_input, y_output)

	def reset(self):
		self.bucket = []
		self.line_counter = 0

	def fill_bucket(self):
		print("ayyy")
		q = 0
		temp_buffer = []
		f_in = open(self.path,'r')
		num_col = 7
		item_count = 0
		while(True):
			while(q < self.item_count):
				t_count = 0
				while(t_count < 7):
					f_in.readline()
					t_count += 1
				q+= 1
			count = 0
			data_buff = np.zeros((7,60))
			while(count < 7):
				line = f_in.readline()
				if line == "" or line == "\n" or line == None:
					self.eof = True
					break
				line = line[:-1]
				split = line.split(",")
				a = np.array(split)
				data_buff[count] = a
				count += 1
			if self.eof == True:
				return -1
			self.bucket.append(data_buff)
			self.item_count += 1
			item_count += 1
			if item_count == 1000:
				break

		return 0


#testing stuff yea only 1 satelite reads somethihng at a pparticular time. WHich makes sense since they are moving
# def seperate_by_satelite(f_path = './data/fff_Bw_1min.DAT'):
# 	f_in = open(f_path,'r')
# 	sat_1 = open('./data/Sat_1.DAT')
# 	sat_2 = open('./data/Sat_2.DAT')
# 	sat_3 = open('./data/Sat_3.DAT')

# 	while(True):
# 		line = f_in.readline()
# 		if line == "" or line == None or line == "\n":
# 			break
# 	return 0

def chunk(f_path = './data/fff_Bw_1min.DAT'):
	f_in =  open(f_path,'r')
	f_out = open("./data/ConvData.DAT",'w')
	useful_col = [5,6,7,8,9,10]
	num_channel = len(useful_col)+1
	buff = np.zeros((7,60))
	output_col = [11]
	line_count = 0
	incomplete = False
	while(True):
		row_count = 0
		while(row_count < 60):
			print(line_count)
			line = f_in.readline()
			line_count += 1
			if line == "\n" or line == "" or line == None:
				incomplete = True
				break
			line = line[:-1]

			split = line.split(" ")
			split = (list)(filter(lambda a: a != '', split))
			#correct for potential mis-spacing between 6 and 7
			if len(split) != 12:
				a = split[6]
				a = a.split('-')
				del split[6]
				split.insert(6,a[0])
				split.insert(7,"-"+a[1])
			useful = split[5:]

			col_count = 0
			while(col_count < num_channel):
				if col_count == (num_channel - 1):
					a = float(useful[-1].replace(',','.')) * 100000
					buff[col_count][row_count] = a
				else:
					buff[col_count][row_count] = (float)(useful[col_count])
				col_count += 1
			row_count += 1
		if incomplete == True:
			break
		list_form = buff.tolist()
		for i in range(num_channel):
			stri = (str)(list_form[i][0])
			j = 1
			while( j < len(list_form[0])):
				stri = stri + ',' + (str)(list_form[i][j])
				j += 1
			print(stri,file=f_out)
	f_in.close()
	f_out.close()
	print("Done converting files")

#a = ConvDataset(1,'idk','./data/ConvData.DAT')
#X,Y = a.__getitem__()
#print(X)
#print(Y)
#chunk()