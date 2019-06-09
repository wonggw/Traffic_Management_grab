import pickle
import numpy as np
import torch
import time
from pykalman import UnscentedKalmanFilter

timing=[]
with open("traffic_image_fake.txt", "rb") as fp:

	b = pickle.load(fp)


for i in range(len(b[0])):
	print (b[0][i])
	time.sleep(1)
