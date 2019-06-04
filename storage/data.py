import pandas as pd
import numpy as np
import operator
import Geohash
import pickle

days=[]
timestamps=[]
geohash=[]
demands=[]
images=[]
timing=[]

print("Loading data ......")
df=pd.read_csv(filepath_or_buffer = "training.csv")
print("Processing data ......")
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M').dt.time
df.sort_values(["day","timestamp"], axis = 0, ascending = [True,True], inplace = True) 

print (df)

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
			timing.append((timestamp.hour * 60 + timestamp.minute)/15)
			images.append(image)
			image= np.zeros(shape=(25,5)).tolist()
			j=j+1
			#print (timing)
		day=row['day']
		timestamp=row['timestamp']
		image[int((-(float(coordinates[0])+5.24))*100)][int((float(coordinates[1])-90.6)*10)]=row['demand']

	else:
		image[int((-(float(coordinates[0])+5.24))*100)][int((float(coordinates[1])-90.6)*10)]=row['demand']
		#print(image)

images.append(image)
timing.append((timestamp.hour * 60 + timestamp.minute)/15)
#print(images)
#print (np.asarray(images).shape)

with open("traffic_image.txt", "wb") as fp:

	pickle.dump([images,timing], fp)

