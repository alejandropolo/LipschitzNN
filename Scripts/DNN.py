import torch.nn as nn
from collections import OrderedDict
import torch.nn.init as init

class DNN(nn.Module):
    def __init__(self, layers, activations):
        super(DNN, self).__init__()
        
        ## Check lenght of layers and activations
        if len(layers)-1 < len(activations):
            raise Exception('The number of activations must be equal or less than the number of layers')

        # Parameters
        self.depth = len(layers) - 1
        
        # Deploy layers with activations
        layer_list = []
        for i in range(self.depth):
            layer_list.append(('layer_%d' % i, nn.Linear(layers[i], layers[i+1])))
            # Determine the activation function for this layer
            if i < len(activations):
                activation_function = activations[i].lower()
            elif i == self.depth - 1:
                activation_function = None
            else:
                activation_function = 'relu'

            # Create the activation layer based on the specified activation function
            if activation_function == 'relu':
                activation = nn.ReLU()
            elif activation_function == 'sigmoid':
                activation = nn.Sigmoid()
            elif activation_function == 'tanh':
                activation = nn.Tanh()
            elif activation_function == 'leakyrelu':
                activation = nn.LeakyReLU()
            elif activation_function is None:
                activation = None
            else:
                raise Exception(f'Invalid activation function: {activation_function}')
            
            if activation is not None:
                layer_list.append(('activation_%d' % i, activation))
        
        
        layer_dict = OrderedDict(layer_list)
        
        # Deploy layers
        self.layers = nn.Sequential(layer_dict)
        
        # He Initialization for linear layers
        ################## TO-DO: IMPLEMENTAR HE INITIALIZATION PARA TODAS LAS ACTIVACIONES ##################
        for name, module in self.layers.named_children():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='sigmoid')
        
        
    def forward(self, x):
        out = self.layers(x)
        return out