# import pytorch libraries
import torch
from torch import nn


# define encoder class
class Encoder(nn.Module):

    # define class constructor
    def __init__(self, input_size, hidden_size):

        # call super class constructor
        super(Encoder, self).__init__()

        # init individual layers
        self.map1_L = nn.Linear(input_size, hidden_size[0], bias=True)
        nn.init.xavier_uniform_(self.map1_L.weight)
        nn.init.constant_(self.map1_L.bias, 0.0)
        self.map1_R = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        self.map2_L = nn.Linear(hidden_size[0], hidden_size[1], bias=True)
        nn.init.xavier_uniform_(self.map2_L.weight)
        nn.init.constant_(self.map2_L.bias, 0.0)
        self.map2_R = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        self.map3_L = nn.Linear(hidden_size[1], hidden_size[2], bias=True)
        nn.init.xavier_uniform_(self.map3_L.weight)
        nn.init.constant_(self.map3_L.bias, 0.0)
        self.map3_R = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        self.map4_L = nn.Linear(hidden_size[2], hidden_size[3], bias=True)
        nn.init.xavier_uniform_(self.map4_L.weight)
        nn.init.constant_(self.map4_L.bias, 0.0)
        self.map4_S = torch.nn.Sigmoid()

    # define forward pass
    def forward(self, x):

        # run forward pass
        x = self.map1_R(self.map1_L(x))
        x = self.map2_R(self.map2_L(x))
        x = self.map3_R(self.map3_L(x))
        x = self.map4_S(self.map4_L(x))

        # return result
        return x
