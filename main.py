import torch
import argparse
import time  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import teacherNet,Net, cifar_Net, cnn_Net
from densenet import densenet_cifar
from teacher_tr import teacher_train, teacher_train_eval , teacher_test
from torchvision import datasets, transforms
from loss import distillation
from resnet import ResNet18
import os 



parser = argparse.ArgumentParser(description='knowledge distillation ex')
parser.add_argument('--data', type=str, default='cifar10', metavar='N',
                    help='input batch size for training (default: 64)') 
parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                    help='SGD weight_decay (default: 0.5)')
#parser.add_argument('--no-cuda', action='store_true', default=False,
#                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--type', type=str, default='1',
                    help='how many batches to wait before logging training status')
parser.add_argument('--temp', type=int, default=20, metavar='N',
                    help='temperature scaling')
parser.add_argument('--alpha', type=int, default=0.9, metavar='N',
                    help='alpha')
parser.add_argument('--mode', type=str, default='train', metavar='N',
                    help='alpha')
parser.add_argument('--gpu',type=str,default='0',
                    help = 'gpu')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

if args.data == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data_cifar', train=True, download=True,
                            transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                    ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data_cifar', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                    ])),
        batch_size=args.batch_size, shuffle=False)
elif args.data == 'mnist': 
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data_mnist', train=True, download=True,
                            transform=transforms.Compose([
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data_mnist', train=False, download=True,
                            transform=transforms.Compose([
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=False)




#train mode 
if args.mode == 'train':
    if args.data == 'cifar10':
    # models 
        if args.model == 'resnet18' : 
            teacher_model = ResNet18().cuda()
        elif args.model == 'densenet':    
            teacher_model = densenet_cifar().cuda()
        elif args.model == 'mlp' : 
            teacher_model = teacherNet().cuda()    
        elif args.model == 'cnn' :
            teacher_model = cnn_Net(num_channels=32).cuda()
    elif args.data == 'mnist': 
        if args.model == 'resnet18' : 
            teacher_model = ResNet18().cuda()
            teacher_model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda() 
            
        elif args.model == 'densenet':    
            teacher_model = densenet_cifar().cuda()
        elif args.model == 'mlp' : 
            teacher_model = teacherNet().cuda()    
        elif args.model == 'cnn' :
            teacher_model = cnn_Net(num_channels=32).cuda()         
            
            

    optimizer = optim.SGD(teacher_model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=150, gamma=0.1)

    for epoch in range(1, args.epochs + 1): 
        
            scheduler.step() 
            st = time.time()
            teacher_train(epoch = epoch, model = teacher_model,optimizer = optimizer,train_loader =  train_loader,log_interval = args.log_interval)
            teacher_train_eval(model = teacher_model, train_loader = train_loader)
            teacher_test(model = teacher_model, test_loader = test_loader) 
            print("train time %d seconds:" %(time.time()-st))
    if args.data == 'mnist':        
        if args.model == 'resnet18' :           
            torch.save(teacher_model.state_dict(), 'mnist_teacher_resnet %f.pth'%(args.weight_decay))
        if args.model == 'densenet' :   
            torch.save(teacher_model.state_dict(), 'mnist_teacher_densenet %f.pth'%(args.weight_decay))  
        if args.model == 'cnn' :   
            torch.save(teacher_model.state_dict(), 'mnist_teacher_cnn %f.pth'%(args.weight_decay))      
    elif args.data == 'cifar10' :         
        if args.model == 'resnet18' :           
            torch.save(teacher_model.state_dict(), 'teacher_resnet %f.pth'%(args.weight_decay))
        if args.model == 'densenet' :   
            torch.save(teacher_model.state_dict(), 'teacher_densenet %f.pth'%(args.weight_decay))  
        if args.model == 'cnn' :   
            torch.save(teacher_model.state_dict(), 'teacher_cnn %f.pth'%(args.weight_decay))    
    
    
# distill to student network
elif args.mode == 'kd':
    if args.model == 'resnet18' : 
        teacher_model = ResNet18().cuda()
        teacher_model.load_state_dict(torch.load('teacher_resnet %f.pth'%(args.weight_decay)),strict=False) 
    elif args.model == 'mlp' : 
        teacher_model = teacherNet().cuda()
    elif args.model == 'densenet' : 
        teacher_model = densenet_cifar().cuda()
        teacher_model.load_state_dict(torch.load('teacher_densenet %f.pth'%(args.weight_decay)),strict=False) 

    if (args.data == 'cifar10') & (args.type == '1') :  
        student_model = cifar_Net().cuda()
    elif (args.data == 'cifar10') & (args.type == '2') :
        student_model = cnn_Net(num_channels=32).cuda()    
    else :
        student_model = Net().cuda()
    optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch, teacher_model,model,loss_fn):
    model.train()
    teacher_model.eval()
    st = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        with torch.no_grad():
            teacher_output = teacher_model(data)
            teacher_output = teacher_output.detach()
        # teacher_output = Variable(teacher_output.data, requires_grad=False) #alternative approach to load teacher_output
        loss_fn = distillation(output, target, teacher_output, T=args.temp, alpha=args.alpha).cuda()
        loss_fn.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_fn.data))
    print("kd train time %d seconds:" %(time.time()-st)) 
    
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        # test_loss += F.cross_entropy(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    #test_loss /= len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.6f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch = epoch, teacher_model = teacher_model,model = student_model, loss_fn=distillation)
    test(model = student_model)

torch.save(student_model.state_dict(), 'distill.pth.tar')






#import pdb; pdb.set_trace()