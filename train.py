from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
import numpy as np

from src.resnet import ResNet18

def build_loader_file_paths(pert_size, sample_count):

    train_tensors = f'./attacks/train/{sample_count}_{pert_size}'
    test_tensors = f'./attacks/test/{sample_count}_{pert_size}'
    ind_tensors = f'./attacks/ind/{sample_count}_{pert_size}'

    return train_tensors, test_tensors, ind_tensors

def poison_training_data(trainset, images, labels, ind_train):
    image_dtype = trainset.data.dtype
    
    images = np.rint(np.transpose(images.numpy() * 255, [0, 2, 3, 1])).astype(image_dtype)
    trainset.data = np.concatenate((trainset.data, images))
    trainset.targets = np.concatenate((trainset.targets, labels))

    trainset.data = np.delete(trainset.data, ind_train, axis=0)
    trainset.targets = np.delete(trainset.targets, ind_train, axis=0)
    return trainset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training (with backdoor)')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument("--strength", default=1, type=int, help="[1, 2, 4, 8, 16, 32, 64, 128, 255]")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Load in attack data
if not os.path.isdir('attacks'):
    print('Attack images not found, please craft attack images first!')
    sys.exit(0)

train_atk_path, test_atk_path, ind_atk_path = build_loader_file_paths(args.strength, 500)

# train_attacks = torch.load('./attacks/train_attacks')
train_attacks = torch.load(train_atk_path)
train_images_attacks = train_attacks['image']
train_labels_attacks = train_attacks['label']
# test_attacks = torch.load('./attacks/test_attacks')
test_attacks = torch.load(test_atk_path)
test_images_attacks = test_attacks['image']
test_labels_attacks = test_attacks['label']

# Normalize backdoor test images
testset_attacks = torch.utils.data.TensorDataset(test_images_attacks, test_labels_attacks)

ind_train = torch.load(ind_atk_path)
trainset = poison_training_data(trainset, train_images_attacks, train_labels_attacks, ind_train)

# Load in the datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
attackloader = torch.utils.data.DataLoader(testset_attacks, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()


def lr_scheduler(epoch):
    lr = 1e-3
    if epoch > 65:
        lr *= 1e-3
    elif epoch > 55:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)

    return lr


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_scheduler(epoch))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print('Train ACC: %.3f' % acc)

    return net


# Test
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print('Test ACC: %.3f' % acc)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        best_acc = acc


# Test ASR
def test_attack(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(attackloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print('Attack success rate: %.3f' % acc)


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+70):
        train(epoch)
        test(epoch)
        test_attack(epoch)

