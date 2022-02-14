import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from da import _DA
from dataset import trainset

train_data  = trainset(['dataset/_train_S{}.csv'.format(i) for i in ['805', '809', '814']])
trainloader = DataLoader(train_data, batch_size=1, shuffle=True)

class Net(nn.Module):
    def __init__(self, input_dim=8, middle=5):
        super(Net, self).__init__()
        self.input_dim = input_dim 
        self.middle = middle 
        self.L1 = nn.Linear(input_dim, middle, bias=True)
        self.L2 = nn.Linear(middle, 1, bias=True)
        self.reLu = nn.ReLU(inplace=False)
        self.mse_loss = torch.nn.MSELoss(reduce=False)

    def forward(self, x, y):
        x = self.L2(self.reLu(self.L1(x)))
        mse = self.mse_loss(x, y)
        return mse