from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import torch
import glob
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from PIL import Image
from torchvision.datasets import ImageFolder

class cifar10(Dataset):
    def __init__(self, train):
        super(cifar10, self).__init__()
        self.transform = Compose([
            ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])

        if train:
            ds = CIFAR10(root='../data', train=True, download=True)
        else:
            ds = CIFAR10(root='../data', train=False, download=True)

        self.data = ds.data
        self.targets = ds.targets
        self.classes = ds.classes
        self.class_to_idx = ds.class_to_idx
        self.zoc_splits =  [[0, 1, 9, 7, 3, 2],
                            [0, 2, 4, 3, 7, 5],
                            [5, 1, 9, 8, 7, 0],
                            [5, 7, 1, 8, 4, 6],
                            [8, 1, 5, 3, 4, 6]
                           ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])

class cifar100(Dataset):
    def __init__(self, train):
        super(cifar100, self).__init__()
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    
        self.transform = Compose([
            ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])

        if train:
            ds = CIFAR100(root='../data', train=True, download=True)
        else:
            ds = CIFAR100(root='../data', train=False, download=True)
        
        self.data = ds.data
        self.targets = ds.targets
        self.classes = ds.classes
        self.class_to_idx = ds.class_to_idx
        self.zoc_splits = [list(range(20)), list(range(20, 40)), list(range(40, 60)), list(range(60, 80)), list(range(80, 100))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])

def data_loader(dataset, train):
    if dataset == 'cifar10':
        dataset = cifar10(train)
    elif dataset == 'cifar100':
        dataset = cifar100(train)

    loader = DataLoader(dataset=dataset, batch_size=128, num_workers=4)
    return loader

def tinyimage_semantic_split_generator():
   index_splits = [
       [192, 112, 145, 107, 91, 180, 144, 193, 10, 125, 186, 28, 72, 124, 54, 77, 157, 169, 104, 166],
       [156, 157, 167, 175, 153, 11, 147, 0, 199, 171, 132, 60, 87, 190, 101, 111, 193, 71, 131, 192],
       [28, 15, 103, 33, 90, 167, 61, 13, 124, 159, 49, 12, 54, 78, 82, 107, 80, 25, 140, 46],
       [128, 132, 123, 72, 154, 35, 86, 10, 188, 28, 85, 89, 91, 82, 116, 65, 96, 41, 134, 25],
       [102, 79, 47, 106, 59, 93, 145, 10, 62, 175, 76, 183, 48, 130, 38, 186, 44, 8, 29, 26]]  # CAC splits
   dataset = ImageFolder(root='../data/tiny-imagenet-200/val')
   a=dataset.class_to_idx
   reverse_a = {v:k for k,v in a.items()}
   semantic_splits = [[],[],[],[],[]]
   for i, split in enumerate(index_splits):
       wnid_split = []
       for idx in split:
           wnid_split.append(reverse_a[idx])
       all = list(dataset.class_to_idx.keys())
       seen = wnid_split
       unseen = list(set(all)-set(seen))
       #seen.extend(unseen)
       f = open('./imagenet_utils/imagenet_id_to_label.txt', 'r')
       imagenet_id_idx_semantic = f.readlines()

       for id in seen:
           for line in imagenet_id_idx_semantic:
               if id == line[:-1].split(' ')[0]:
                   semantic_label = line[:-1].split(' ')[2]
                   semantic_label = semantic_label.replace("_", " ")
                   semantic_splits[i].append(semantic_label)
                   break
   return index_splits, semantic_splits

class tinyimagenet(Dataset):
    def __init__(self, train, mappings_dict):
        super(tinyimagenet, self).__init__()
        self.index_splits, self.semantic_splits = tinyimage_semantic_split_generator()
        self.image_paths = []
        self.targets = []

        # run imagenet_utils/tinyimagenet.sh to download tinyimagenet data
        if train == True:
            path = '../data/tiny-imagenet-200/train/'
        else:
            path = '../data/tiny-imagenet-200/val/'
        for label in mappings_dict.keys():
            temp_path = path + f"{mappings_dict[label]}"
            self.image_paths.extend(glob.glob(os.path.join(temp_path,'*.JPEG')))
            self.targets.extend(len(glob.glob(os.path.join(temp_path,'*.JPEG')))*[label])
        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        self.classes = list(mappings_dict.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform:
            x = self.transform(x)
        return x

def tinyimage_loader(train):
    
    f = open('./imagenet_utils/tinyimagenet_labels_to_ids.txt', 'r')
    #f = open('../tinyimagenet_ids_to_label.txt', 'r')
    tinyimg_label2folder = f.readlines()
    mappings_dict = {}
    for line in tinyimg_label2folder:
        label, class_id = line[:-1].split(' ')[0], line[:-1].split(' ')[1]
        label = label.replace("_", " ")
        mappings_dict[label] = class_id

    dataset = tinyimagenet(train, mappings_dict)
    loader = DataLoader(dataset=dataset, batch_size=128, num_workers=4)
    return loader
