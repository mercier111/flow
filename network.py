import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

#from da import _DA
from dataset import trainset


class Net(nn.Module):
    def __init__(self, input_dim=7, middle=10, m2=5):
        super(Net, self).__init__()
        self.input_dim = input_dim 
        self.middle = middle 
        self.L1 = nn.Linear(input_dim, middle, bias=True)
        self.L2 = nn.Linear(middle, m2, bias=True)
        self.L3 = nn.Linear(m2, 1, bias=True)
        self.reLu = nn.ReLU(inplace=False)
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, x, y):
        x = self.reLu(self.L1(x))
        x = self.reLu(self.L2(x))
        x = self.reLu(self.L3(x))
        x = torch.squeeze(x)
        mse = self.mse_loss(x, y)  # boardcast issue !!!
        return x, mse


#mse_loss = torch.nn.MSELoss(reduction='sum')
#x = torch.tensor([1., 2.])
#y = torch.tensor([2., 1.])
#print(mse_loss(x, y))