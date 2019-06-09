import pickle
import numpy as np
import torch
from pykalman import UnscentedKalmanFilter

images=[]
with open("traffic_image.txt", "rb") as fp:

	b = pickle.load(fp)


def cal_tensor(x):
	num_image=torch.from_numpy(np.asarray([x]))
	c=(num_image[0][i]+2*num_image[0][i+1]+2*num_image[0][i+2]+num_image[0][i+3])/6
	return c.tolist()

images.append(b[0][0])
images.append(b[0][1])
for i in range(len(b[0])-3):
	c=cal_tensor(b[0])
	images.append(c)
	images.append(b[0][i+2])
	#print (i)

with open("traffic_image_fake.txt", "wb") as fp:

	pickle.dump([images], fp)
