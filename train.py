import torch 
import time 
import os 

from torch.utils.data import DataLoader
from torch.autograd import Variable
from model.utils.net_utils import clip_gradient, save_checkpoint

from dataset import trainset
from network import Net

#################  settings  #############
batch_size = 100
max_epochs = 5
cuda = True 
disp_interval = 50000
max_iter = 250000
lr = 0.01
iters_per_epoch = max_iter
checkpoint_interval = 1
output_dir = "save/"
#########################################


train_data  = trainset(['dataset/_train_S{}.csv'.format(i) for i in ['805', '809', '814']])
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

print('train data : {} '.format(len(train_data)))
net = Net(middle=8)

#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0)

input = torch.FloatTensor(7)
output = torch.FloatTensor(1)
if cuda:
    input = input.cuda()
    output = output.cuda()
    net.cuda()

input = Variable(input)
output = Variable(output)

for epoch in range(1, max_epochs + 1):
    loss_temp = 0
    start = time.time()
    
    data_iter = iter(trainloader)
    for step in range(iters_per_epoch):
        try:
            data = next(data_iter)
        except:
            data_iter = iter(trainloader)
            data = next(data_iter)

        input.resize_(data[0].size()).copy_(data[0])
        output.resize_(data[1].size()).copy_(data[1])
        

        net.zero_grad()
        mse_loss = net(input, output)

        loss = mse_loss
        loss_temp += loss.item()

        optimizer.zero_grad()
        loss.backward()
        #clip_gradient(net, 100)
        optimizer.step()

        if step % disp_interval == 0:
            end = time.time()
            if step > 0:
                loss_temp /= disp_interval + 1
            
            mse_loss = mse_loss.item()

            print( "[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                        % (epoch, step, max_iter, loss_temp, lr)
                    )
            print( "\t\t\ttime_cost: %.4f"
                        % (end - start)
                    )
            print( "\t\t\tmse_loss: %.4f"
                        % (mse_loss)
                    )

            loss_temp = 0
            start = time.time()

    if epoch % checkpoint_interval == 0 or epoch == max_epochs:
        save_name = os.path.join(
            output_dir, "{}.pth".format('flow' + "_" + str(epoch)),
        )
        save_checkpoint(
            {
                "iter": step + 1,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            save_name,
        )
        print("save model: {}".format(save_name))