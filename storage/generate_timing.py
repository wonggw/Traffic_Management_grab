import pickle
import numpy as np
import torch
from pykalman import UnscentedKalmanFilter

timing=[]
with open("traffic_image.txt", "rb") as fp:

	b = pickle.load(fp)

#print (b[0][0])

def one_hot(x):
	num_time=torch.from_numpy(np.asarray([x]))
	num_time=torch.zeros(1, 96).scatter_(1, num_time.unsqueeze(1), 1.)
	return num_time

timing.append(one_hot(b[1][0]).tolist())
timing.append(one_hot(b[1][1]).tolist())
for i in range(len(b[1])-3):


	c=(np.asarray(one_hot(b[1][i+1])+one_hot(b[1][i+2]))/2)
	timing.append(c.tolist())
	timing.append(one_hot(b[1][i+2]).tolist())
with open("traffic_timing_fake.txt", "wb") as fp:

	pickle.dump([timing], fp)
