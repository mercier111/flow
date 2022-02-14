import torch 
import time 
from torch.utils.data import DataLoader
from torch.autograd import Variable


from dataset import trainset
from network import Net


batch_size = 200
max_epochs = 15
cuda = True 
disp_interval = 5000
max_iter = 10000
lr = 0.01
iters_per_epoch = 100

train_data  = trainset(['dataset/_train_S{}.csv'.format(i) for i in ['805', '809', '814']])
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

net = Net(middle=5)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

input = torch.FloatTensor(1)
output = torch.FloatTensor(1)
if cuda:
    input = input.cuda()
    output = output.cuda()
    net.cuda()

input = Variable(input)
output = Variable(output)

for epoch in range(max_epochs + 1):
    loss_temp = 0
    start = time.time()
    
    data = iter(train_data)
    for step in range(iters_per_epoch):
            try:
                data = next(train_data)
            except:
                train_data = iter(train_data)
                data = next(train_data)
    net.zero_grad()
    mse_loss = net(input, output)

    loss = mse_loss
    loss_temp += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if step % disp_interval == 0:
        end = time.time()
        if step > 0:
            loss_temp /= disp_interval + 1
        mse_loss = mse_loss.item()
        print( "[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                    % (epoch, step, max_iter, loss_temp, lr)
                )
        print( "\t\t\tmse_loss: %.4f"
                    % (mse_loss)
                )

        loss_temp = 0
        start = time.time()