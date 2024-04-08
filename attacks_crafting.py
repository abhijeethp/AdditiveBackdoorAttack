from __future__ import absolute_import
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import json
import numpy as np

from src.utils import pattern_craft, add_backdoor

SC, TC = 8, 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed()

# Load raw data
print('==> Preparing data..')
transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

num_perturbed_samples = 500
for pert_strength in [1, 2, 4, 8, 16, 32, 64, 128, 255]:

    perturbation = pattern_craft(trainset.__getitem__(0)[0].size(), (pert_strength / 255))

    # Crafting training backdoor images
    train_images_attacks = None
    train_labels_attacks = None
    ind_train = [i for i, label in enumerate(trainset.targets) if label==SC]
    ind_train = np.random.choice(ind_train, int(num_perturbed_samples), False)
    for i in ind_train:
        if train_images_attacks is not None:
            train_images_attacks = torch.cat([train_images_attacks, add_backdoor(trainset.__getitem__(i)[0], perturbation).unsqueeze(0)], dim=0) # (TO DO)
            train_labels_attacks = torch.cat([train_labels_attacks, torch.tensor([TC], dtype=torch.long)], dim=0)
        else:
            train_images_attacks = add_backdoor(trainset.__getitem__(i)[0], perturbation).unsqueeze(0)
            train_labels_attacks = torch.tensor([TC], dtype=torch.long)

    # Crafting test backdoor images
    test_images_attacks = None
    test_labels_attacks = None
    ind_test = [i for i, label in enumerate(testset.targets) if label==SC]
    for i in ind_test:
        if test_images_attacks is not None:
            test_images_attacks = torch.cat([test_images_attacks, add_backdoor(testset.__getitem__(i)[0], perturbation).unsqueeze(0)], dim=0)
            test_labels_attacks = torch.cat([test_labels_attacks, torch.tensor([TC], dtype=torch.long)], dim=0)
        else:
            test_images_attacks = add_backdoor(testset.__getitem__(i)[0], perturbation).unsqueeze(0)
            test_labels_attacks = torch.tensor([TC], dtype=torch.long)

    # Create attack dir and save attack images
    if not os.path.isdir('attacks'):
        os.mkdir('attacks')
    for _dir in ["train", "test", "ind"]:
        if not os.path.isdir(f'attacks/{_dir}'):
            os.mkdir(f'attacks/{_dir}')

    train_attacks = {'image': train_images_attacks, 'label': train_labels_attacks}
    test_attacks = {'image': test_images_attacks, 'label': test_labels_attacks}

    torch.save(train_attacks, f'./attacks/train/{num_perturbed_samples}_{pert_strength}')
    torch.save(test_attacks, f'./attacks/test/{num_perturbed_samples}_{pert_strength}')
    torch.save(ind_train, f'./attacks/ind/{num_perturbed_samples}_{pert_strength}')
