from comet_ml import Experiment 
import torch
import argparse
import time  
import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import random
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from model import teacherNet,Net, cifar_Net, cnn_Net
from teacher_tr import teacher_train, teacher_train_eval , teacher_test
from torchvision import datasets, transforms
from loss import distillation
from resnet import ResNet18
from wrn import Wide_ResNet
from gaussian_noise import add_noise
import os 

#scratch


parser = argparse.ArgumentParser(description='knowledge distillation ex')
parser.add_argument('--data', type=str, default='cifar10', metavar='N',
                    help='data') 
parser.add_argument('--ood_data', type=str, default='None', metavar='N',
                    help='ood')                     
parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                    help='model')
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
parser.add_argument('--temp', type=int, default=10, metavar='N',
                    help='temperature scaling')
parser.add_argument('--alpha', type=int, default=0.9, metavar='N',
                    help='alpha')
parser.add_argument('--mode', type=str, default='train', metavar='N',
                    help='alpha')
parser.add_argument('--method', type=str, default='None', metavar='N',
                    help='method')
parser.add_argument('--noise_variance', type=float, default=0.1, metavar='N',
                    help='noise_variance')
parser.add_argument('--noise_label', type=float, default=0.1, metavar='N',
                    help='noise_rate(messycollab)') 
parser.add_argument('--gpu',type=str,default='0',
                    help = 'gpu')

args = parser.parse_args()

experiment = Experiment(project_name='project',
                        api_key='api')
experiment.log_parameters(args)



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
            teacher_model = models.ResNet18().cuda()
        elif args.model == 'wrn' : 
            teacher_model = models.Wide_ResNet(depth=40, num_classes=10, widen_factor=2,dropout_rate=0.0).cuda()
        elif args.model == 'wrn_s' :
            teacher_model = models.Wide_ResNet(depth=16, num_classes=10, widen_factor=2,dropout_rate=0.0).cuda()         
    elif args.data == 'mnist': 
        if args.model == 'resnet18' : 
            teacher_model = models.ResNet18().cuda()
            teacher_model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda() 
      
    print(teacher_model)        
       

    optimizer = optim.SGD(teacher_model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay,nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[60,120,150], gamma=0.2)
    
    for epoch in range(1, args.epochs + 1): 
        
            scheduler.step() 
            st = time.time()
            teacher_train(epoch = epoch, model = teacher_model,optimizer = optimizer,train_loader =  train_loader,log_interval = args.log_interval,method=args.method,noise_variance=args.noise_variance)
            teacher_train_eval(model = teacher_model, train_loader = train_loader)
            teacher_test(model = teacher_model, test_loader = test_loader) 
            print("train time %d seconds:" %(time.time()-st))

    if args.data == 'mnist':        
        if args.model == 'resnet18' :           
            torch.save(teacher_model.state_dict(), 'mnist_teacher_resnet %f.pth'%(args.weight_decay))      
    elif args.data == 'cifar10' :         
        if args.model == 'resnet18' :           
            torch.save(teacher_model.state_dict(), 'teacher_resnet %f.pth'%(args.weight_decay)) 
        if args.model == 'wrn' : 
            torch.save(teacher_model.state_dict(), 'wrn_40_2 %f.pth'%(args.weight_decay))

        if args.model == 'wrn_s' : 
            torch.save(teacher_model.state_dict(), 'wrn_16_2 %f.pth'%(args.weight_decay))


elif args.mode == 'kd':
    if args.model == 'resnet18' : 
        teacher_model = ResNet18().cuda()
        teacher_model.load_state_dict(torch.load('teacher_resnet %f.pth'%(args.weight_decay)),strict=False) 
    elif args.model == 'wrn' : 
        teacher_model = Wide_ResNet(depth=40, num_classes=10, widen_factor=2,dropout_rate=0.0).cuda()
        teacher_model.load_state_dict(torch.load('wrn_40_2 %f.pth'%(args.weight_decay)),strict=False)

    #load student network     

    if (args.data == 'cifar10') & (args.type == '1') : 
        student_model = Wide_ResNet(depth=16, num_classes=10, widen_factor=2,dropout_rate=0.0).cuda()      

    optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[60,120,150], gamma=0.2)

def train(epoch, teacher_model,model,loss_fn):
    with experiment.train():

        scaler = GradScaler(enabled=True)

        step = 0
        scheduler.step() 
        model.train()
        teacher_model.eval()
        st = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            with autocast(enabled=True) : 
                with torch.no_grad() :
                    teacher_output = teacher_model(data)
                    teacher_output = teacher_output.detach()
                
                if args.method == 'softrandom' :  
                    data = add_noise(data,args.noise_variance)
                output = model(data)
            
            
        
            
            if args.method == 'messycollab' : 
                for i in range(int(args.noise_label * data.shape[0])):
                    perturbation = random.randint(0, data.shape[1] - 1)
                    if perturbation != target[i]:
                        target[i] = perturbation
                    else:
                        try:
                            perturbation + 1 < data.shape[1]
                            target[i] = perturbation + 1
                        except:
                            target[i] = perturbation - 1 

            optimizer.zero_grad()


            loss_fn = distillation(output, target, teacher_output, T=args.temp, alpha=args.alpha).cuda()
            scaler.scale(loss_fn).backward()
            step += 1
            experiment.log_metric('train_loss',loss_fn)
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_fn.data))
        print("kd train time %d seconds:" %(time.time()-st)) 
    
def test(model):
    with experiment.test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        experiment.log_metric('test_accuracy', (100. * correct / len(test_loader.dataset)) )
        print('\nTest set: Accuracy: {}/{} ({:.6f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
   
def ood_test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad() : 
        ood_test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./data_SVHN', split='test', download=True,
                            transform=transforms.Compose([
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
                    ])),
        batch_size=args.batch_size, shuffle=False)
        for data, target in ood_test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            
            pred = output.data.max(1, keepdim=True)[1] 
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('\nTest set: Accuracy: {}/{} ({:.6f}%)\n'.format(
            correct, len(ood_test_loader.dataset),
            100. * correct / len(ood_test_loader.dataset)))
 
for epoch in range(1, args.epochs + 1):
    print(args.method, args.noise_variance)
    train(epoch = epoch, teacher_model = teacher_model,model = student_model, loss_fn=distillation)
    if args.ood_data =='svhn' : 
        print('ood_test')
        ood_test(model = student_model)
    else : 
        test(model = student_model)
torch.save(student_model.state_dict(), 'distill.pth.tar')






#import pdb; pdb.set_trace()