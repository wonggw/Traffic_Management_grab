import pickle

with open("traffic_image.txt", "rb") as fp:

	b = pickle.load(fp)

print (b)
