import torch.nn as nn
import torch.nn.functional as F
import torch
torch.manual_seed(0)
class ExponentialLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ExponentialLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, input):
        return F.linear(input, torch.exp(self.weight), self.bias)

######################## CLASE CON PESOS EXPONENCIALES

class MLP_Exponential(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP_Exponential, self).__init__()
        self.linear1 = ExponentialLinear(in_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.linear2 = ExponentialLinear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x
