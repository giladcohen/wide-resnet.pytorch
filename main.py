from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime
import numpy as np
import scipy.ndimage

from networks import *
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--root_dir', default='/data/gilad/logs/log_XXXX', type=str, help='path to root dir')
parser.add_argument('--data_dir', default='/data/dataset/cifar10', type=str, help='path to data dir')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--image', help='Input an image')
parser.add_argument('--var_loss', action='store_true', help='Use the new variance loss')
args = parser.parse_args()

checkpoint_dir = os.path.join(args.root_dir, 'checkpoint')

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
total_cm = np.zeros((10, 10))  # just for CIFAR-10. Must be updated for CIFAR-100
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, int(num_classes), args.var_loss)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

net, file_name = getNetwork(args)
checkpoint_file = os.path.join(checkpoint_dir, args.dataset, file_name+'.t7')

# Input an image for testing
if (args.image):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir(checkpoint_dir), 'Error: No checkpoint directory found!'
    _, file_name = net, file_name
    checkpoint = torch.load(checkpoint_file)
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()

    if use_cuda:
        inputs = scipy.ndimage.imread(args.image).cuda()
    inputs = torch.from_numpy(inputs)
    inputs = Variable(inputs, volatile=True)
    print(inputs)
    outputs = net(inputs)
    print(outputs)

    predicted = torch.max(outputs.data, 1)

    print("| Test Result: ", predicted)

    sys.exit(0)

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir(checkpoint_dir), 'Error: No checkpoint directory found!'
    _, file_name = net, file_name
    checkpoint = torch.load(checkpoint_file)
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_se = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        # Confusion Matrix
        cm = confusion_matrix(y_true=targets.data, y_pred=predicted)
        total_cm += cm
        # RMSE
        err = targets.data - predicted
        total_se.extend(err * err)
    rmse = np.sqrt(np.mean(total_se))

    acc = 100.*correct/total
    print("| Test Result\tAcc@1: %.2f%%" %(acc))

    print("RMSE:\n", "0.08828386397")
    print("Confusion Matrix:\n", total_cm)

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir(checkpoint_dir), 'Error: No checkpoint directory found!'
    _, file_name = net, file_name
    checkpoint = torch.load(checkpoint_file)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = net, file_name
    net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def criterion2(outputs1, outputs2):
    batch_size = outputs1.shape[0]
    out1 = outputs1.view(batch_size, -1)
    out2 = outputs2.view(batch_size, -1)
    std1 = out1.std(dim=1)
    std2 = out2.std(dim=1)
    # l1_loss = nn.L1Loss()
    # loss = l1_loss(std1, std2)
    loss = torch.sum((std1 - std2)**2) / std1.data.nelement()
    return loss

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4, nesterov=True)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs, outputs1, outputs2 = net(inputs)               # Forward Propagation
        loss1 = criterion(outputs, targets)  # Loss
        loss2 = criterion2(outputs1, outputs2)
        loss = loss1 + 1.0 * loss2
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.data[0], 100.0*correct/total))
        sys.stdout.flush()

def test(epoch):
    global best_acc
    global cm
    global total_cm
    global rmse
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_se = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets)
        outputs, outputs1, outputs2 = net(inputs)
        loss1 = criterion(outputs, targets)
        loss2 = criterion2(outputs1, outputs2)
        loss = loss1 + 1.0 * loss2

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()
        # Confusion Matrix
        cm = confusion_matrix(y_true=targets.data, y_pred=predicted)
        total_cm += cm
        # RMSE
        err = targets.data - predicted
        total_se.extend(err * err)
    rmse = np.sqrt(np.mean(total_se))

    # Save checkpoint when best model
    acc = 100.0*correct/total

    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))
    cm = confusion_matrix(y_true=targets.data, y_pred=predicted)
    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':net.module if use_cuda else net,
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_point = os.path.join(checkpoint_dir, args.dataset)
        if not os.path.isdir(save_point):
            os.makedirs(save_point)
        torch.save(state, os.path.join(save_point, file_name+'.t7'))
        best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))

print("RMSE:\n", rmse)
print("Confusion Matrix:\n", total_cm)
