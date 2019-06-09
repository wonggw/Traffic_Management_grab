import torch
from torch.nn import functional as F
from torch import nn

class AddCoords(nn.Module):

	def __init__(self, with_r=False):
		super(AddCoords,self).__init__()
		self.with_r = with_r

	def forward(self, input_tensor):
		"""
		Args:
			input_tensor: shape(batch, channel, x_dim, y_dim)
		"""
		batch_size, _, x_dim, y_dim = input_tensor.size()

		xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
		yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

		xx_channel = xx_channel.float() / (x_dim - 1)
		yy_channel = yy_channel.float() / (y_dim - 1)

		xx_channel = xx_channel * 2 - 1
		yy_channel = yy_channel * 2 - 1

		xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
		yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

		ret = torch.cat([
			input_tensor,
			xx_channel.type_as(input_tensor),
			yy_channel.type_as(input_tensor)], dim=1)

		if self.with_r:
			rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
			ret = torch.cat([ret, rr], dim=1)

		return ret


class NAC_conv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding,init_fun=nn.init.xavier_uniform_):
		super(NAC_conv2d,self).__init__()

		self.padding=padding
		self.conv_W = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
		self.conv_M = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

		self._W_hat = self.conv_W.weight
		self._M_hat = self.conv_M.weight

		self.register_parameter('W_hat', self._W_hat)
		self.register_parameter('M_hat', self._M_hat)

		torch.nn.init.xavier_uniform_(self.conv_W.weight)
		torch.nn.init.xavier_uniform_(self.conv_M.weight)

	def forward(self, x):
		W = torch.tanh(self._W_hat) * torch.sigmoid(self._M_hat)
		return F.conv2d(x,W, padding=self.padding)



class NALU_conv2d(nn.Module):
	def __init__(self,  in_channels, out_channels, kernel_size, padding,init_fun=nn.init.xavier_uniform_):
		super(NALU_conv2d,self).__init__()
		self.padding=padding
		self.conv_G = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
		self._G = self.conv_G.weight
		self.register_parameter('G', self._G)
		torch.nn.init.xavier_uniform_(self.conv_G.weight)

		self._nac = NAC_conv2d(in_channels, out_channels, kernel_size, padding, init_fun=init_fun)

		self._epsilon = 1e-8

		self.addcoords = AddCoords(with_r=False)

	def forward(self, x):
		x=self.addcoords(x)
		g = torch.sigmoid(F.conv2d(x,self._G, padding=self.padding))
		m = torch.exp(self._nac(torch.log(torch.abs(x) + self._epsilon)))
		a = self._nac(x)

		y = g * a + (1 - g) * m

		return y


class NAC_conv3d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding,init_fun=nn.init.xavier_uniform_):
		super(NAC_conv3d,self).__init__()

		self.padding=padding
		self.conv_W = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
		self.conv_M = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

		self._W_hat = self.conv_W.weight
		self._M_hat = self.conv_M.weight
		self.register_parameter('W_hat', self._W_hat)
		self.register_parameter('M_hat', self._M_hat)

		torch.nn.init.xavier_uniform_(self.conv_W.weight)
		torch.nn.init.xavier_uniform_(self.conv_M.weight)

	def forward(self, x):
		W = torch.tanh(self._W_hat) * torch.sigmoid(self._M_hat)
		return F.conv3d(x,W, padding=self.padding)



class NALU_conv3d(nn.Module):
	def __init__(self,  in_channels, out_channels, kernel_size, padding,init_fun=nn.init.xavier_uniform_):
		super(NALU_conv3d,self).__init__()
		self.padding=padding
		self.conv_G = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
		self._G = self.conv_G.weight
		self.register_parameter('G', self._G)
		torch.nn.init.xavier_uniform_(self.conv_G.weight)

		self._nac = NAC_conv3d(in_channels, out_channels, kernel_size, padding, init_fun=init_fun)

		self._epsilon = 1e-8

	def forward(self, x):
		g = torch.sigmoid(F.conv3d(x,self._G, padding=self.padding))

		m = torch.exp(self._nac(torch.log(torch.abs(x) + self._epsilon)))

		a = self._nac(x)

		y = g * a + (1 - g) * m

		return y


