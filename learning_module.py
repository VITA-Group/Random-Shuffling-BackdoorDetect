import os
import sys
import torch
import numpy as np

from dataset.patch_based_cifar10 import PatchedCIFAR10
from dataset.clean_label_cifar10 import CleanLabelPoisonedCIFAR10
from dataset.poisoned_gtsrb import PoisonedGTSRB
from dataset.multiple_targets import PatchedCIFAR10_mt

from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from models.adv_resnet import resnet20s as robust_res20s
from models.resnets import *
from models.preactresnet import *
from models.alexnet import *

num_workers = 0

def get_loader(args, rev=False):
    if args.dataset.lower() == "cifar10":
        if (args.atk == 'cla' or args.atk == 'patch') and not args.multiple_targets:
            if args.atk == 'cla':
                print('Clean Label Attack')
                robust_model = robust_res20s(num_classes = 10)
                robust_weight = torch.load(args.robust_model, map_location='cpu')

                if 'state_dict' in robust_weight.keys():
                    robust_weight = robust_weight['state_dict']
                
                keys=[]
                for k,v in robust_weight.items():
                    if k.startswith('normalize'):
                        continue
                    keys.append(k)

                new_robust_weight = {k:robust_weight[k] for k in keys}
                
                robust_model.load_state_dict(new_robust_weight)
                train_set = CleanLabelPoisonedCIFAR10(args.data+'/cifar10', poison_ratio=args.rate, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, source=args.source, black_trigger=not args.color, robust_model=robust_model, augmentation=True, use_normalize=False)
            
            elif args.atk == 'patch':
                print('BadNets')
                train_set = PatchedCIFAR10(args.data+'/cifar10', mode='train', poison_ratio=args.rate, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, source=args.source, black_trigger=not args.color, augmentation=True, use_normalize=False)
                
            poison_valset = PatchedCIFAR10(args.data+'/cifar10',mode='val', poison_ratio=1, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, source=args.source, black_trigger=not args.color, use_normalize=False)
            poison_valloader = DataLoader(poison_valset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
            
            poison_testset = PatchedCIFAR10(args.data+'/cifar10',mode='test', poison_ratio=1, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, source=args.source, black_trigger=not args.color, use_normalize=False)
            poison_testloader = DataLoader(poison_testset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)

        elif args.atk == 'wanet':
            train_set = PatchedCIFAR10(args.data+'/cifar10', mode='train', poison_ratio=0, augmentation=False, use_normalize=False)
            poison_valloader, poison_testloader = None, None
        
        elif args.multiple_targets and args.target_num:
            train_set = PatchedCIFAR10_mt(args.data+'/cifar10', mode='train', poison_ratio=args.rate, target_num=args.target_num, patch_size=5, augmentation=True, use_normalize=False)
            poison_valset = PatchedCIFAR10_mt(args.data+'/cifar10', mode='val', target_num=args.target_num, patch_size=5, use_normalize=False)
            poison_testset = PatchedCIFAR10_mt(args.data+'/cifar10', mode='test', target_num=args.target_num, patch_size=5, use_normalize=False)
            poison_valloader = DataLoader(poison_valset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
            poison_testloader = DataLoader(poison_testset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)

        else:
            print("attack not yet implemented")
            sys.exit()

        clean_valset = PatchedCIFAR10(args.data+'/cifar10', mode='val', poison_ratio=0, use_normalize=False)
        clean_testset = PatchedCIFAR10(args.data+'/cifar10', mode='test', poison_ratio=0, use_normalize=False)

        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        valloader = DataLoader(clean_valset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        testloader = DataLoader(clean_testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    elif args.dataset.lower() == 'gtsrb':
        if args.atk == 'patch':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
            train_set = PoisonedGTSRB(args.data, train=True, transform=transform, poison_ratio=args.rate, patch_size=args.patch_size,
                random_loc=False, target=0, color=args.color)

            train_subset = np.random.choice(list(range(len(train_set))), size=int(len(train_set)*0.9), replace=False)
            val_subset = np.setdiff1d(list(range(len(train_set))), train_subset)

            train_set = Subset(train_set, train_subset)

            poi_train_set = PoisonedGTSRB(args.data, train=True, transform=transform, poison_ratio=1, patch_size=args.patch_size,
                random_loc=False, target=0, color=args.color)
            clean_train_set = PoisonedGTSRB(args.data, train=True, transform=transform, poison_ratio=0, patch_size=args.patch_size,
                random_loc=False, target=0, color=args.color)
            poi_valset = Subset(poi_train_set, train_subset)
            clean_valset = Subset(clean_train_set, val_subset)
            
            clean_testset = PoisonedGTSRB(args.data, train=False, transform=transform, poison_ratio=0, patch_size=args.patch_size,
                random_loc=False, target=0, color=args.color)
            poison_testset = PoisonedGTSRB(args.data, train=False, transform=transform, poison_ratio=1, patch_size=args.patch_size,
                random_loc=False, target=0, color=args.color)
            
            trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
            valloader = DataLoader(clean_valset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            poison_valloader = DataLoader(poi_valset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            testloader = DataLoader(clean_testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            poison_testloader = DataLoader(poison_testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    else:
        print("Dataset not yet implemented")
        sys.exit()
    print('finish loading dataset')

    return trainloader, valloader, poison_valloader, testloader, poison_testloader
        

def get_model(args, contain_normalize=True):

    model_name = args.model.lower()
    if model_name == 'resnet20s':
        return resnet20s(args.num_classes, contain_normalize=contain_normalize)
    elif model_name == 'preactresnet18':
        return PreActResNet18(args.num_classes, contain_normalize=contain_normalize)
    elif model_name == 'alexnet':
        return alexnet(num_classes=args.num_classes, contain_normalize=contain_normalize)

