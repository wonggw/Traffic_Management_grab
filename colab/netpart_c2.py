import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nalu
from torch.autograd import Variable
import preprocessing_c
from torchvision import datasets, transforms


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.time_sequence=8
		self.conv1 = nn.Conv2d(1, 4, kernel_size=1, padding=0)
		self.conv1_bn = nn.BatchNorm2d(5)
		self.conv2 = nn.Conv2d(5, 3, kernel_size=3, padding=1)
		self.conv2_bn = nn.BatchNorm2d(8)

		self.gru1 = nn.GRUCell(1000+96, 512)
		self.gru2 = nn.GRUCell(512, 512)
		self.gru3 = nn.GRUCell(512, 512)
		self.gru4 = nn.GRUCell(512, 512)
		self.gru5 = nn.GRUCell(512, 512)
		self.gru6 = nn.GRUCell(512, 512)

		self.nalu2=nalu.NALU(512,125)

		torch.nn.init.xavier_uniform_(self.conv1.weight)
		torch.nn.init.xavier_uniform_(self.conv2.weight)

	def init_hidden(self,device,batch_size):


		# the weights are of the form (nb_layers, batch_size, nb_lstm_units)
		hidden_a = torch.zeros([int(batch_size), 512], dtype=torch.double).to(device)
		hidden_b = torch.zeros([int(batch_size), 256], dtype=torch.double).to(device)
		#hidden_b = torch.FloatTensor(3, int(batch_size), 1000).uniform_(0.5, -0.5).to(device)


		hidden_a= Variable(hidden_a)
		hidden_b= Variable(hidden_b)
		#hidden_b = Variable(hidden_b)

		return hidden_a.double(),hidden_b.double()

	def forward(self, x,timing,device):
		output=[]
		x = x.view(-1,1,25,5)

		batch_size=int((x.size()[0])/self.time_sequence)
		hidden_a,hidden_b = self.init_hidden(device,batch_size)
		hn1=hidden_a
		hn2=hidden_a
		hn3=hidden_a
		hn4=hidden_a
		hn5=hidden_a
		hn6=hidden_a

		x1 = F.elu(self.conv1(x))
		x1=self.conv1_bn(torch.cat((x, x1), 1))
		x2= F.elu(self.conv2(x1))
		x=self.conv2_bn(torch.cat((x1, x2), 1))
		x = x.view(int(batch_size*self.time_sequence), -1)
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
			hn4=self.gru4(hn3,hn4)
			if i<self.time_sequence-1:
				hn4=hn3+hn4
			hn5=self.gru5(hn4,hn5)
			if i<self.time_sequence-1:
				hn5=hn4+hn5
			hn6=self.gru6(hn5,hn6)

		x = self.nalu2(hn6)
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
epochs=1000

def main():
	dataset=preprocessing_c.Traffic_Dataset()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1, shuffle=True)


	model = Net().double().to(device)
	loss_fn = nn.MSELoss(reduction='mean')
	optimizer = optim.RMSprop(model.parameters(), lr=0.0001)


	#model.load_state_dict(torch.load("drive/Colab/grab/Traffic_management/model/traffic_cnn.pt"))
	#model.to(device)

	for epoch in range(1, epochs + 1):
		train( model, device, train_loader, loss_fn,optimizer, epoch)
		torch.save(model.state_dict(),"drive/Colab/grab/Traffic_management/model/traffic_cnn.pt")

	#torch.save(model.state_dict(),"traffic_cnn.pt")

if __name__ == '__main__':
	main()



