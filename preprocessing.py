import torch
import pandas as pd
import numpy as np
import Geohash
import pickle

days=[]
timestamps=[]
geohash=[]
demands=[]
images=[]


class Traffic_Dataset():

	def __init__(self):
		print("Loading data ......")
		#images=self.read_data()
		with open("./storage/traffic_image.txt", "rb") as fp:
			saved_data = pickle.load(fp)
		images=saved_data[0]
		timing=torch.from_numpy(np.asarray(saved_data[1]))
		
		self.timing=saved_data[1]
		self.time_sequence=8
		self.len=np.asarray(images).shape[0]
		self.x_data=images

	def __getitem__(self,index):
		num_images=[]
		num_time=[]
		num_target=[]
		for i in range(self.time_sequence):
			num_images.append(self.x_data[index+i])
			num_time.append(self.timing[index+i])
			if i>(self.time_sequence-2):
				num_target.append(self.x_data[index+i+1])
		num_time=(torch.from_numpy(np.asarray(num_time)).double())/95
		num_target=torch.from_numpy(np.asarray(num_target))
		return torch.from_numpy(np.asarray(num_images)),num_target,num_time

	def __len__(self):
		#print("Size of data",self.len)
		return (self.len-self.time_sequence)

	def read_data(self):

		df=pd.read_csv(filepath_or_buffer = "training.csv")
		print("Processing data ......")
		df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M')
		df.sort_values(["day","timestamp"], axis = 0, ascending = [True,True], inplace = True) 
		#print (df)

		image= np.zeros(shape=(25,5)).tolist()
		#print(image)
		i=-1
		j=0
		for index, row in df.iterrows():
			#print(row["day"],row["timestamp"])
			coordinates=Geohash.decode(row['geohash6'])
			i=i+1
			if i==0 or day!=row['day'] or timestamp!=row['timestamp']:
				if i!=0:
					images.append(image)
					image= np.zeros(shape=(25,5)).tolist()
					j=j+1
				day=row['day']
				timestamp=row['timestamp']
				image[int((-(float(coordinates[0])+5.24))*100)][int((float(coordinates[1])-90.6)*10)]=row['demand']

			else:
				image[int((-(float(coordinates[0])+5.24))*100)][int((float(coordinates[1])-90.6)*10)]=row['demand']
				#print(image)

		#print(images)
		#print (np.asarray(images).shape)
		return images


#	if i==0:
#		min_x=coordinates[0]
#		max_x=coordinates[0]
#		min_y=coordinates[1]
#		max_y=coordinates[1]
#	else:
#		if coordinates[0]<min_x:
#			min_x=coordinates[0]
#		if coordinates[0]>max_x:
#			max_x=coordinates[0]
#		if coordinates[1]<min_y:
#			min_y=coordinates[1]
#		if coordinates[1]>max_y:
#			max_y=coordinates[1]
	#print(row['day'],coordinates)

#dataset=Traffic_Dataset()
#dataset.__getitem__(5)
#dataset.__len__()
