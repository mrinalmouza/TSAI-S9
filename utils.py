import torch 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pdb
import albumentations as A


# train_transform = transforms.Compose([transforms.ToTensor(),
#                                       transforms.Normalize((0.49,0.48,0.44), 
#                                                            (0.24,0.24,0.261)) 
#                                       ])
# test_transform = transforms.Compose([transforms.ToTensor(),
#                                      transforms.Normalize((0.49,0.48,0.45), 
#                                                            (0.24,0.24,0.26))
#                                      ])

coarse_dropout_augmentation = A.augmentations.dropout.coarse_dropout.CoarseDropout (
                                           max_holes=8, max_height=16, max_width=16, min_holes=8, min_height=16, 
                                           min_width=16, fill_value=[0.4914, 0.4822, 0.4465], mask_fill_value=None, always_apply=False, p=0.2)

custom_transform = transforms.Lambda(lambda img: coarse_dropout_augmentation(image=np.array(img))['image'])
# Train Phase transformations
train_transform = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(15),
                                       transforms.ToTensor(),
                                       #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])

# Test Phase transformations
test_transform = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                       ])

# train_transform = transforms.Compose([transforms.ToTensor()
                                       
#                                       ])
# test_transform = transforms.Compose([transforms.ToTensor()
                                     
#                                      ])

# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # extra transfrom for the training data, in order to achieve better performance
# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     
# ])

train_data  = datasets.CIFAR10('../data', train = True, download= True, transform= train_transform)
test_data   = datasets.CIFAR10('../data', train = False, download= True, transform= test_transform)

dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True)

# train dataloader
train_loader = torch.utils.data.DataLoader(train_data,**dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)


