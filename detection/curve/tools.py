import os
import sys
import torch
import numpy as np
import random
import torchvision
from torch.utils.data import DataLoader, Subset
sys.path.append("../..")
from dataset.patch_based_cifar10 import PatchedCIFAR10
from dataset.clean_label_cifar10 import CleanLabelPoisonedCIFAR10
from models.adv_resnet import resnet20s as robust_res20s
from dataset.poisoned_gtsrb import PoisonedGTSRB
from dataset.multiple_targets import PatchedCIFAR10_mt

from trainer.wanet.utils import *
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.stats import norm

def more_config(args, print_log=True):

    if args.atk == 'patch':
        args.model_name = 'color' if args.color else 'black'
    elif args.atk == 'cla':
        args.model_name = 'cla'
    elif args.atk == 'wanet':
        args.model_name = 'k_%d_s_%.1f'%(args.k, args.s)
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.atk == 'patch':
        if args.random_loc and args.bottom_left:
            print('Specify the location')
            sys.exit()
        elif not args.random_loc and not args.bottom_left:
            args.upper_right = True
        else:
            args.upper_right = False
    
    elif args.atk == 'cla':
        if args.random_loc and args.bottom_left:
            print('Specify the location')
            sys.exit()
        elif not args.random_loc and not args.bottom_left:
            args.upper_right = True
        else:
            args.upper_right = False


    if args.dataset.lower() == 'cifar10':
        args.num_classes = 10
        args.input_height, args.input_width, args.input_channel = 32, 32, 3
    elif args.dataset.lower() == 'gtsrb':
        args.num_classes = 43

    if args.model == 'ResNet20s':
        args.N_layer, args.feat_dim = 19, 64
    elif args.model == 'PreActResNet18' or args.model == 'ResNet18':
        args.N_layer, args.feat_dim = 17, 512
    elif args.model == 'alexnet':
        args.N_layer, args.feat_dim = 5, 1024

def get_loader(args):
    
    if args.dataset.lower() == 'cifar10' and not args.multiple_targets :
        if args.atk == 'cla':
            print('Clean Label Attack')
            robust_model = robust_res20s(num_classes = args.num_classes)
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
            train_set = PatchedCIFAR10(args.data+'/cifar10', mode='train', poison_ratio=args.rate, patch_size=args.patch_size,
                                random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                target=args.target, source=args.source, black_trigger=not args.color, augmentation=True, use_normalize=False)
        elif args.atk == 'wanet':
            train_set = PatchedCIFAR10(args.data+'/cifar10', mode='train', poison_ratio=0, augmentation=False, use_normalize=False)

        trainset_sample = random.sample(range(45000), args.data_num)
        train_set = Subset(train_set, trainset_sample)
        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
    
    if args.dataset.lower() == 'cifar10' and args.multiple_targets:
        train_set = PatchedCIFAR10_mt(args.data+'/cifar10', mode='train', poison_ratio=args.rate, target_num=args.target_num, patch_size=5, augmentation=False, use_normalize=False)
        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    elif args.dataset.lower() == 'gtsrb':
        if args.atk == 'patch':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
            train_set = PoisonedGTSRB(args.data, train=True, transform=transform, poison_ratio=args.rate, patch_size=args.patch_size,
                random_loc=False, target=0, color=args.color)

            train_subset = np.random.choice(list(range(len(train_set))), size=args.data_num, replace=False)

            train_set = Subset(train_set, train_subset)
            trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            
    return trainloader

def test_shuffle_patch(args, model, dataloader, shuffle):
    model.eval()
    natural_correct, total = 0, 0

    features = torch.empty([args.data_num, args.feat_dim])
    labels = torch.empty([args.data_num])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):

            if batch_idx * args.batch_size >= args.data_num:
                break
            
            if shuffle:
                model = shuffle_ckpt_layer(model, args.shuffle_index, type=args.model=='alexnet')

            total += inputs.shape[0]

            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feature = model(inputs)
            _, natural_predicted = outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()

            features[batch_idx*args.batch_size: min((batch_idx+1)*args.batch_size, len(dataloader.dataset))] = feature
            labels[batch_idx*args.batch_size: min((batch_idx+1)*args.batch_size, len(dataloader.dataset))] = targets
    
    features_save = {}
    for c in range(args.num_classes):
        idx = torch.nonzero(torch.eq(labels, c)).squeeze()
        features_save['class_%d'%c] = features[idx]
                
    natural_acc = 100.0 * natural_correct / total
    print('ACC: %.2f'%natural_acc)

    return features_save

def test_shuffle_wanet(args, model, dataloader, noise_grid, identity_grid, shuffle):

    model.eval()

    total_clean = 0
    total_bd = 0
    total_cross = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0

    transforms = PostTensorTransform(args).to(args.device)

    features = torch.empty([args.data_num, args.feat_dim])
    labels = torch.empty([args.data_num])

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):

            if batch_idx * args.batch_size >= args.data_num:
                break
            if shuffle:
                model = shuffle_ckpt_layer(model, args.shuffle_index)

            x, y = x.to(args.device), y.to(args.device)
            bs = x.shape[0]
            num_bd = int(bs * args.rate)
            num_cross = int(num_bd * args.cross_ratio)

            # generate true backdoor data
            grid_temps = (identity_grid + args.s * noise_grid / args.input_height) * args.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            # noise mode
            ins = torch.rand(num_cross, args.input_height, args.input_height, 2).to(args.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / args.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = F.grid_sample(x[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
            targets_bd = torch.ones_like(y[:num_bd]) * args.target

            inputs_cross = F.grid_sample(x[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)

            total_inputs = torch.cat([inputs_bd, inputs_cross, x[(num_bd + num_cross) :]], dim=0)
            total_inputs = transforms(total_inputs.cpu()).to(args.device)

            total_targets = torch.cat([targets_bd, y[num_bd:]], dim=0)

            total_preds, feature = model(total_inputs)
            features[batch_idx*args.batch_size: min((batch_idx+1)*args.batch_size, len(dataloader.dataset))] = feature
            labels[batch_idx*args.batch_size: min((batch_idx+1)*args.batch_size, len(dataloader.dataset))] = total_targets

            total_clean += bs - num_bd - num_cross
            total_bd += num_bd
            total_cross += num_cross
            total_clean_correct += torch.sum(torch.argmax(total_preds[(num_bd + num_cross) :], dim=1) == total_targets[(num_bd + num_cross) :])
            try:
                total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
            except: 
                pass
            try:
                total_cross_correct += torch.sum(
                    torch.argmax(total_preds[num_bd : (num_bd + num_cross)], dim=1) == total_targets[num_bd : (num_bd + num_cross)])
            except:
                pass

    avg_acc_clean = total_clean_correct * 100.0 / total_clean
    try:
        avg_acc_bd = total_bd_correct * 100.0 / total_bd
    except:
        avg_acc_bd = 0
    try:
        avg_acc_cross = total_cross_correct * 100.0 / total_cross
    except:
        avg_acc_cross = 0
    
    print('clean: %.2f, backdoor: %.2f, cross: %.2f'%(avg_acc_clean, avg_acc_bd, avg_acc_cross))

    features_save = {}
    for c in range(args.num_classes):
        idx = torch.nonzero(torch.eq(labels, c)).squeeze()
        features_save['class_%d'%c] = features[idx]

    return features_save

def shuffle_ckpt_layer(model, shuffle_index, type=False):
    model_state = model.state_dict()
    new_ckpt = {}
    i = 0
    for k, v in model_state.items():
        if 'conv' in k or (type and len(v.shape)==4):
            if shuffle_index[i] == 1:
                _, channels, _, _ = v.size()

                idx = torch.randperm(channels)
                v = v[:,idx,...]
            i += 1
        new_ckpt[k] = v
    model_state.update(new_ckpt)
    model.load_state_dict(model_state, strict=True)
    return model

def mad(X, seed, name=None, draw=False):
    X1 = X - torch.mean(X, dim=1, keepdim=True)
    
    median = torch.median(X, dim=0, keepdim=True)[0]
    median_1 = torch.median(X1, dim=0, keepdim=True)[0]

    X = torch.norm(X-median, dim=1, p=1).numpy()
    X1 = torch.std(X1-median_1, dim=1).numpy()

    X = X1 + 0.01 * X

    med = np.median(X, axis=0)
    abs_dev = np.absolute(X - med)
    med_abs_dev = np.median(abs_dev)

    mod_z_score = norm.ppf(0.75) * abs_dev / med_abs_dev
    result = (mod_z_score) * (X > med)
    return result