import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nalu
import conv_nalu
import conv_gru
from torch.autograd import Variable
import preprocessing_c
from torchvision import datasets, transforms


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.time_sequence=18
		self.conv1 = conv_nalu.NALU_conv2d(3, 15, kernel_size=3, padding=1)
		self.conv1_bn = nn.BatchNorm2d(16)
		self.conv2 =conv_nalu.NALU_conv2d(18, 32, kernel_size=3, padding=1)
		self.conv2_bn = nn.BatchNorm2d(48)
		self.conv3 =conv_nalu.NALU_conv2d(50, 1, kernel_size=1, padding=0)
		self.conv3_bn = nn.BatchNorm2d(2)

		self.conv1_gru=conv_gru.ConvGRUCell(2,48, kernel_size=3)
		self.conv2_gru=conv_gru.ConvGRUCell(48,48, kernel_size=1)
		self.conv3_gru=conv_gru.ConvGRUCell(48,48, kernel_size=3)
		self.conv4_gru=conv_gru.ConvGRUCell(48,48, kernel_size=1)
		self.conv5_gru=conv_gru.ConvGRUCell(48,48, kernel_size=3)
		self.conv6_gru=conv_gru.ConvGRUCell(48,48, kernel_size=1)

		self.conv4 =conv_nalu.NALU_conv2d(50, 12, kernel_size=3, padding=1)
		self.conv4_bn = nn.BatchNorm2d(12)
		self.conv1_simple = nn.Conv2d(12, 1, kernel_size=1, padding=0)
	def init_hidden(self,device,batch_size):


		# the weights are of the form (nb_layers, batch_size, nb_lstm_units)
		hidden_a = torch.ones([int(batch_size), 125+96], dtype=torch.double).to(device)
		hidden_b = torch.ones([int(batch_size), 125], dtype=torch.double).to(device)
		#hidden_b = torch.FloatTensor(3, int(batch_size), 1000).uniform_(0.5, -0.5).to(device)

		hidden_a= Variable(hidden_a)
		hidden_b= Variable(hidden_b)
		#hidden_b = Variable(hidden_b)

		return hidden_a.double(),hidden_b.double()

	def forward(self, x,timing,device):
		output=[]
		hn1,hn2,hn3,hn4,hn5,hn6=[None for _ in range(6)]
		batch_size=int(x.size()[0])
		timing_new_channel=torch.empty([1, 25,5], dtype=torch.double).to(device)

		combined_list=torch.empty([1,int(batch_size),2, 25,5], dtype=torch.double).to(device)

		x=x.unsqueeze(2)
		x=x.transpose(0,1) # time,batch,channel,width,height
		timing=timing.transpose_(0, 1)
		for i in range(self.time_sequence):
			timing_list=torch.empty([1, 25,5], dtype=torch.double).to(device)
			x1 = self.conv1(x[i])
			x1=self.conv1_bn(torch.cat((x[i], x1), 1))
			x2=self.conv2(x1)
			x3= (torch.cat((x1, x2), 1))
			x3=self.conv2_bn(x3)
			x4=self.conv3(x3)
			#combined= torch.stack(x5+combine)
			for j in range(len(timing[i])):
				timing_new_channel.fill_(timing[i][j])
				timing_list=torch.cat((timing_list,timing_new_channel),0)
			timing_list=timing_list[1:]
			timing_list=timing_list.unsqueeze(1)
			combined=torch.cat((x4,timing_list),1)
			combined=self.conv3_bn(combined)
			combined=combined.unsqueeze(0)
			combined_list=torch.cat((combined_list,combined),0)

		x=combined_list[1:]


		for i in range(self.time_sequence):
			hn1=self.conv1_gru(x[i],hn1,device)
			#if i<self.time_sequence-1:
			#	hn1=x[i]+hn1
			hn2=self.conv2_gru(hn1,hn2,device)
			if i<self.time_sequence:
				hn2=hn1+hn2
			hn3=self.conv3_gru(hn2,hn3,device)
			if i<self.time_sequence:
				hn3=hn2+hn3
			hn4=self.conv4_gru(hn3,hn4,device)
			if i<self.time_sequence:
				hn4=hn3+hn4
			hn5=self.conv5_gru(hn4,hn5,device)
			if i<self.time_sequence:
				hn5=hn4+hn5
			hn6=self.conv6_gru(hn5,hn6,device)
			if i<self.time_sequence:
				hn6=hn5+hn6

		x=self.conv4_bn(self.conv4(hn6))
		x=self.conv1_simple(x)
		x=x.view(-1,125)
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
		clipping_value = 1	#arbitrary number of your choosing
		torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
		optimizer.step()
		if batch_idx % 50 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))
epochs=1000

def main():
	dataset=preprocessing_c.Traffic_Dataset()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device =="cuda":
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1, shuffle=True)


	model = Net().double().to(device)
	loss_fn = nn.MSELoss(reduction='mean')
	optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)


	model.load_state_dict(torch.load("drive/Colab/grab/Traffic_management/model/traffic_cnn.pt"))
	model.to(device)

	for epoch in range(1, epochs + 1):
		#if epoch %20==0:
		#	optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0)
		#else:
		#	optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
		train( model, device, train_loader, loss_fn,optimizer, epoch)
		torch.save(model.state_dict(),"drive/Colab/grab/Traffic_management/model/traffic_cnn.pt")

	#torch.save(model.state_dict(),"traffic_cnn.pt")

if __name__ == '__main__':
	main()



