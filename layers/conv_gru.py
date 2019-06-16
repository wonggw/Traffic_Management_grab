# Inspired by Alfredo Canziani (http://tinyurl.com/CortexNet/)

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
from torch.autograd import Variable

class ConvGRUCell(nn.Module):
	"""
	Generate a convolutional GRU cell
	"""

	def __init__(self, input_size, hidden_size, kernel_size):
		super(ConvGRUCell,self).__init__()
		padding = kernel_size // 2
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
		self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
		self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

		init.orthogonal_(self.reset_gate.weight)
		init.orthogonal_(self.update_gate.weight)
		init.orthogonal_(self.out_gate.weight)
		init.constant_(self.reset_gate.bias, 0.)
		init.constant_(self.update_gate.bias, 0.)
		init.constant_(self.out_gate.bias, 0.)


	def forward(self, input_, prev_state,device):

		# get batch and spatial sizes
		batch_size = input_.data.size()[0]
		spatial_size = input_.data.size()[2:]

		# generate empty prev_state, if None is provided
		if prev_state is None:
			state_size = [batch_size, self.hidden_size] + list(spatial_size)
			prev_state = (Variable(torch.ones(state_size).double()).to(device))/2

		# data size is [batch, channel, height, width]
		stacked_inputs = torch.cat([input_.double(), prev_state], dim=1)
		update = torch.sigmoid(self.update_gate(stacked_inputs))
		reset = torch.sigmoid(self.reset_gate(stacked_inputs))
		out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
		new_state = prev_state * (1 - update) + out_inputs * update

		return new_state
