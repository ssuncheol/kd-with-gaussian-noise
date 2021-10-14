import torch.nn as nn  
import torch.nn.functional as F 
import time 
import random
from gaussian_noise import add_noise


def teacher_train(epoch, model, optimizer,train_loader,log_interval,method,noise_variance,scheduler):
    model.train()
    st = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        if method == 'ga' : 
            data = add_noise(data,noise_variance)
        else : 
            pass  
                                 
        optimizer.zero_grad()
        #import pdb; pdb.set_trace()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
    print('teacher_network train time is {0}s'.format(time.time()-st))
    
    scheduler.step() 

def teacher_train_eval(model,train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        train_loss += F.cross_entropy(output, target).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

def teacher_test(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.cross_entropy(output, target).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
