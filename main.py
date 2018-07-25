from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf
import utils.misc

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import sys
import time
import argparse
import datetime
import numpy as np
import scipy.ndimage
from losses import losses
from networks import *
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--root_dir', default='/data/gilad/logs/log_XXXX', type=str, help='path to root dir')
parser.add_argument('--data_dir', default='/data/dataset/cifar10', type=str, help='path to data dir')
parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.0, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--image', help='Input an image')
parser.add_argument('--loss_criterion', default=1, type=int, help='Which loss to use')
parser.add_argument('--reg1', default=0.0005, type=float, help='regularization_factor for E_loss')
parser.add_argument('--reg2', default=0.0005, type=float, help='regularization_factor for V_loss')
parser.add_argument('--batch_norm', '-bn', action='store_true', help='Use batch normalization in the architecture')
# parser.add_argument('--rescale_ev_loss', '-rescale', action='store_true', help='Rescaling E_loss and V_loss using config')
# parser.add_argument('--optimizer', default='adam', type=str, help='optimizer = [sgd, adam]')

# map of losses:
# 1: my loss: L2 norm between the input STD and output STD, calculated for the entire layer, not over the batch
# 2: Raja's loss, calculating E and Var over the batch and aim for E=0 and Var=1
args = parser.parse_args()
writer = SummaryWriter(log_dir=args.root_dir)
checkpoint_dir = os.path.join(args.root_dir, 'checkpoint')

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
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

total_cm = np.zeros((num_classes, num_classes))
iters_in_epoch = np.ceil(len(trainset)/batch_size)

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
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, int(num_classes), args.batch_norm)
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
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
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
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
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

        if args.dataset == 'cifar10':
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
    if args.dataset == 'cifar10':
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
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion  = nn.CrossEntropyLoss()
utils.misc.print_model_parameters(net)

def regularization(net):
    if args.loss_criterion == 1:
        E_loss, V_loss = losses.criterion1(net)
    elif args.loss_criterion == 2:
        E_loss, V_loss = losses.criterion2(net.named_parameters())
    else:
        raise AssertionError("args.loss_criterion must be within [1:2] but got {}".format(args.loss_criterion))
    losses.rescale_loss_dict(E_loss, cf.e_loss_scales)
    losses.rescale_loss_dict(V_loss, cf.v_loss_scales)

    e_loss = 0.0
    v_loss = 0.0
    for key in E_loss.keys():
        e_loss += E_loss[key]
        v_loss += V_loss[key]
    e_loss *= args.reg1
    v_loss *= args.reg2
    loss = e_loss + v_loss
    return loss, e_loss, v_loss, E_loss, V_loss  # return components for printing

def save_to_tensorboard(loss_dict, prefix, writer, iter):
    """Saving dictionary of loss to tensorboard
    :param loss_dict: the dictionary of losses
    :param prefix: prefix of the names
    :param writer: tensorboard writer
    :param iter: iteration in training
    :return: None
    """
    for key in loss_dict.keys():
        name = prefix + '/' + key
        writer.add_scalar(name, loss_dict[key].item(), iter)

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if optim_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif optim_type == 'ADAM':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        raise AssertionError("Unknown optimizer name: {}".format(optim_type))

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
            net.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs) # Forward Propagation
        loss1 = criterion(outputs, targets)          # Loss
        loss2, e_loss, v_loss, E_loss, V_loss = regularization(net)
        loss = loss1 + loss2

        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        acc = 100.0*correct/total

        iter = (epoch - 1) * iters_in_epoch + batch_idx
        writer.add_scalar('train/accuracy', acc, iter)
        writer.add_scalar('train/loss1', loss1.item(), iter)
        writer.add_scalar('train/loss2', loss2.item(), iter)
        writer.add_scalar('train/loss2_e', e_loss.item(), iter)
        writer.add_scalar('train/loss2_v', v_loss.item(), iter)
        writer.add_scalar('train/loss', loss.item(), iter)
        save_to_tensorboard(net.e_net, 'train/e_net/' , writer, iter)
        save_to_tensorboard(net.v_net, 'train/v_net/' , writer, iter)
        save_to_tensorboard(V_loss   , 'train/V_loss/', writer, iter)
        save_to_tensorboard(E_loss   , 'train/E_loss/', writer, iter)
        save_to_tensorboard(V_loss   , 'train/V_loss/', writer, iter)

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), acc))
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
        outputs = net(inputs)  # Forward Propagation
        loss1 = criterion(outputs, targets)
        loss2, e_loss, v_loss, E_loss, V_loss = regularization(net)
        loss = loss1 + loss2

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        if args.dataset == 'cifar10':
            # Confusion Matrix
            cm = confusion_matrix(y_true=targets.data, y_pred=predicted)
            total_cm += cm
        # RMSE
        err = targets.data - predicted
        total_se.extend(err * err)
    rmse = np.sqrt(np.mean(total_se))

    # Save checkpoint when best model
    acc = 100.0*correct/total
    iter = epoch * iters_in_epoch
    writer.add_scalar('test/accuracy', acc, iter)
    writer.add_scalar('test/loss1', loss1.item(), iter)
    writer.add_scalar('test/loss2', loss2.item(), iter)
    writer.add_scalar('test/loss2_e', e_loss.item(), iter)
    writer.add_scalar('test/loss2_v', v_loss.item(), iter)
    writer.add_scalar('test/loss', loss.item(), iter)
    save_to_tensorboard(net.e_net, 'test/e_net/' , writer, iter)
    save_to_tensorboard(net.v_net, 'test/v_net/' , writer, iter)
    save_to_tensorboard(E_loss   , 'test/E_loss/', writer, iter)
    save_to_tensorboard(V_loss   , 'test/V_loss/', writer, iter)

    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    if args.dataset == 'cifar10':
        cm = confusion_matrix(y_true=targets.data, y_pred=predicted)
    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                # 'net':net.module if use_cuda else net,
                'net':net,
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

if args.dataset == 'cifar10':
    print("Confusion Matrix:\n", total_cm)

# export scalar data to JSON for external processing
writer.export_scalars_to_json(os.path.join(args.root_dir, "all_scalars.json"))
writer.close()
