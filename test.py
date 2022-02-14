import torch 
import time 
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model.utils.net_utils import clip_gradient

from dataset import trainset
from network import Net


batch_size = 5000
max_epochs = 10
cuda = True 
disp_interval = 10000
max_iter = 100000
lr = 0.01
iters_per_epoch = max_iter

test_data  = trainset(['dataset/_test_S{}.csv'.format(i) for i in ['805', '809', '814']])
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
data_iter = iter(testloader)

print('test data : {} '.format(len(test_data)))

net = Net()
net.eval()

for i in range(len(test_data)):
    data = next(data_iter)
    input = data[0]
    label = data[1]