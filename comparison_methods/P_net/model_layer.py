import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Diagonal(nn.Module):
    def __init__(self, units, input_shape, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Diagonal, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.input_dimension = input_shape
        self.kernel_shape = (self.input_dimension, self.units)
        self.n_inputs_per_node = self.input_dimension // self.units
        rows = torch.arange(self.input_dimension)
        cols = torch.arange(self.units)
        cols = cols.repeat(self.n_inputs_per_node)
        self.nonzero_ind = torch.stack((rows, cols), dim=1)
        self.kernel = nn.Parameter(torch.empty(1,self.input_dimension), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(self.units), requires_grad=True) if self.use_bias else None
        if self.kernel_initializer == 'glorot_uniform':
            nn.init.kaiming_uniform_(self.kernel)
        else:
            # Initialize your kernel tensor based on kernel_initializer
            pass
        if self.use_bias:
            if self.bias_initializer == 'zeros':
                nn.init.zeros_(self.bias)
            else:
                # Initialize your bias tensor based on bias_initializer
                pass
    def forward(self, x):
        n_features = x.shape[1]
        # print('kernel', torch.nonzero(self.kernel).shape)
        mult = x * self.kernel
        # print('mult', torch.nonzero(mult).shape)
        mult = mult.view(-1, self.units, self.n_inputs_per_node)
        mult = mult.sum(dim=2)
        # print('mult', torch.nonzero(mult).shape)
        output = mult.view(-1, self.units)
        # print('output', torch.nonzero(output).shape)
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            activation_fn = getattr(nn.functional, self.activation)
            output = activation_fn(output)
        return output


class SparseTF(nn.Module):
    def __init__(self, units, input_shape, map=None, nonzero_ind=None, kernel_initializer='glorot_uniform',
                 activation='tanh', use_bias=True, bias_initializer='zeros'):
        super(SparseTF, self).__init__()
        self.units = units
        self.map = map
        self.nonzero_ind = nonzero_ind
        self.activation = activation
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer
        self.input_dim = input_shape
        if not self.map is None:
            self.map = self.map.astype(np.float32)
        self.nonzero_ind = np.array(np.nonzero(self.map)).T
        self.kernel_shape = (self.input_dim, self.units)
        self.nonzero_count = self.nonzero_ind.shape[0]
        self.kernel_vector = nn.Parameter(torch.empty(1,self.nonzero_count), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(self.units), requires_grad=True) if self.use_bias else None
        if self.kernel_initializer == 'glorot_uniform':
            nn.init.kaiming_uniform_(self.kernel_vector)
        else:
            pass # Initialize your kernel tensor based on kernel_initializer
        if self.use_bias:
            if self.bias_initializer == 'zeros':
                nn.init.zeros_(self.bias)
            else:
                pass  # Initialize your bias tensor based on bias_initializer
    def forward(self, inputs):
        # Calculate output
        nonzero_indices = torch.tensor(self.nonzero_ind, dtype=torch.int64)
        sparse_tensor = torch.sparse.FloatTensor(nonzero_indices.t(), self.kernel_vector[0,:], self.kernel_shape)
        dense_tensor = sparse_tensor.to_dense()
        # print('dense_tensor',dense_tensor.min(),dense_tensor.max())
        output = torch.mm(inputs, dense_tensor)
        # print('output', output.min(), output.max())
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            activation_fn = getattr(nn.functional, self.activation)
            output = activation_fn(output)
        return output


