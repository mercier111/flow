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
load_name = 'save/flow_{}.pth'.format(10)

test_data  = trainset(['dataset/_test_S{}.csv'.format(i) for i in ['805', '809', '814']])
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
data_iter = iter(testloader)

print('test data : {} '.format(len(test_data)))

net = Net(middle=8)
net.eval()

print("load checkpoint %s" % (load_name))
checkpoint = torch.load(load_name, map_location='cpu')
net.load_state_dict(
    {k: v for k, v in checkpoint["model"].items() if k in net.state_dict()}
)

input = torch.FloatTensor(7)
output = torch.FloatTensor(1)
if cuda:
    input = input.cuda()
    output = output.cuda()
    net.cuda()
input = Variable(input)
output = Variable(output)

ave_mse = 0

for i in range(len(data_iter)-1):
    data = next(data_iter)
    input.resize_(data[0].size()).copy_(data[0])
    output.resize_(data[1].size()).copy_(data[1])
    pred, mse_loss = net(input, output)
    ave_mse += mse_loss / batch_size

ave_mse /= len(data_iter)

print('ave mse : %.2e' % ave_mse)
