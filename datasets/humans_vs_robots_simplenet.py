###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Face Dataset
"""
import os

import torchvision
from torchvision import datasets, transforms

import ai8x



def get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((64,64)), #
            #transforms.RandomCrop((64,64), padding=4), #uncomment to data augment
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(degrees=10),
            #transforms.ColorJitter(brightness=0.2),
            #transforms.GaussianBlur((3,3),(0.1,1)), 
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'FaceSpoof', 'train'), transform=train_transform)

                                
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((64,64)), #32
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        test_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'FaceSpoof', 'test'), transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'humans_vs_robots_simplenet',
        'input': (3, 64, 64),
        'output': ('client', 'imposter'),
        'loader': get_datasets,
    },
]
