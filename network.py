import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from da import _DA
from dataset import trainset


class Net(nn.Module):
    def __init__(self, input_dim=7, middle=10):
        super(Net, self).__init__()
        self.input_dim = input_dim 
        self.middle = middle 
        self.L1 = nn.Linear(input_dim, middle, bias=True)
        self.L2 = nn.Linear(middle, 1, bias=True)
        self.reLu = nn.ReLU(inplace=False)
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, x, y):
        x = self.L1(x)
        x = self.reLu(x)
        x = self.L2(x)
        x = torch.squeeze(x)
        mse = self.mse_loss(x, y)  # boardcast issue !!!
        return mse