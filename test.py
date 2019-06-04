import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import nalu
from torch.autograd import Variable
import preprocessing
from torchvision import datasets, transforms


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.time_sequence=8
		self.conv1 = nn.Conv2d(1, 3, kernel_size=1, padding=0,groups=1)
		self.conv1_bn = nn.BatchNorm2d(3)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
		self.conv2_bn = nn.BatchNorm2d(8)

		self.nalu1=nalu.NALU(1500,1000)
		self.gru1 = nn.GRUCell(1000+96, 512)
		self.gru2 = nn.GRUCell(512, 512)
		self.gru3 = nn.GRUCell(512, 512)
		self.nalu2=nalu.NALU(512,125)

		torch.nn.init.xavier_uniform_(self.conv1.weight)
		torch.nn.init.xavier_uniform_(self.conv2.weight)

	def init_hidden(self,device,batch_size):


		# the weights are of the form (nb_layers, batch_size, nb_lstm_units)
		hidden = torch.zeros([int(batch_size), 512], dtype=torch.double).to(device)
		#hidden_b = torch.FloatTensor(3, int(batch_size), 1000).uniform_(0.5, -0.5).to(device)


		hidden= Variable(hidden)
		#hidden_b = Variable(hidden_b)

		return hidden.double()

	def forward(self, x,timing,device):
		output=[]
		x = x.view(-1,1,25,5)

		batch_size=int((x.size()[0])/self.time_sequence)
		hidden = self.init_hidden(device,batch_size)
		hn1=hidden
		hn2=hidden
		hn3=hidden
		hn4=hidden

		x1 = F.elu(self.conv1_bn(self.conv1(x)))
		x1=torch.cat((x, x1), 1)
		x2= F.elu(self.conv2_bn(self.conv2(x1)))
		x=torch.cat((x1, x2), 1)
		x = x.view(int(batch_size*self.time_sequence), -1)

		x = self.nalu1(x)
		x = x.view(self.time_sequence,-1, 1000)
		timing=timing.view(self.time_sequence,-1,96)
		combined=torch.cat((timing.double(), x), dim=2)
		for i in range(self.time_sequence):
			hn1=self.gru1(combined[i],hn1)
			hn2=self.gru2(hn1,hn2)
			if i<self.time_sequence-1:
				hn2=hn1+hn2
			hn3=self.gru3(hn2,hn3)
			if i<self.time_sequence-1:
				hn3=hn2+hn3
		x = F.relu(self.nalu2(hn3))
		return x


def train(model, device, train_loader,loss_fn, optimizer, epoch):
	model.train()
	for batch_idx, (data, target,timing) in enumerate(train_loader):
		data, target,timing = data.to(device), target.to(device),timing.to(device)
		optimizer.zero_grad()
		output = model(data,timing,device)
		target = target.view(-1, 125)
		loss = loss_fn(output, target)
		loss.backward()
		#clipping_value = 5	#arbitrary number of your choosing
		#torch.nn.utils.clip_grad_value_(model.parameters(), clipping_value)
		optimizer.step()
		if batch_idx % 50 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, train_loader,loss_fn, optimizer, epoch):
	model.train()
	for batch_idx, (data, target,timing) in enumerate(train_loader):
		data, target,timing = data.to(device), target.to(device),timing.to(device)
		output = model(data,timing,device)
		target = target.view(-1, 125)
		loss = loss_fn(output, target)
		output=np.reshape(output.data.numpy(),(25,5))
		#print (output)
		print('Loss: {:.6f}'.format( loss.item()))
		time.sleep(1)
		return output


epochs=1000

def main():
	dataset=preprocessing.Traffic_Dataset()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1, shuffle=True)


	model = Net().double().to(device)
	loss_fn = nn.MSELoss(reduction='mean')
	optimizer = optim.SGD(model.parameters(), lr=0.0001,momentum=0.00001)

	model.load_state_dict(torch.load("model/traffic_cnn.pt", map_location=lambda storage, loc: storage))
	model.to(device)

	for epoch in range(1, epochs + 1):
		output=test( model, device, train_loader, loss_fn,optimizer, epoch)
		#torch.save(model.state_dict(),"traffic_cnn.pt")

	#torch.save(model.state_dict(),"traffic_cnn.pt")

if __name__ == '__main__':
	main()





