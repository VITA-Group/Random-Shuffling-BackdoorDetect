import os
import sys
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import time

from learning_module import (
    get_loader,
    get_model, 
)
from trainer.train_patch import *
from trainer.train_wanet import *
from utils import (
    more_config,
    to_log_file,
    adjust_learning_rate
)

def main(args):
    more_config(args)
    model = get_model(args, contain_normalize=not args.no_normalize).to(args.device)
    trainloader, valloader, poison_valloader, testloader, poison_testloader = get_loader(args)

    if args.resume:
        if args.shuffle_load:
            model = load_shuffle_ckpt(args.resume, model)
        else:
            model = load_ckpt(args.resume, model)

            ta, _ = test_patch(model, testloader, args.device, args.shuffle)
            asr, _ = test_patch(model, poison_testloader, args.device, args.shuffle) 
            print(ta, asr)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_ta, best_asr = 0, 0
    best_diff = 0
    ta_all, asr_all, cross_all =[], [], []
    v_ta_all, v_asr_all, v_cross_all =[], [], []

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_schedule, args.lr_factor)

        if args.atk == 'patch' or args.atk == 'cla':
            train_loss, train_acc, _ = train_patch(model, trainloader, optimizer, criterion, args.device, args.shuffle)

            v_ta, _ = test_patch(model, valloader, args.device, False)
            v_asr, _ = test_patch(model, poison_valloader, args.device, False) 

            ta, _ = test_patch(model, testloader, args.device, False)
            asr, _ = test_patch(model, poison_testloader, args.device, False) 

            to_log_file('Epoch %3d: Loss: %.3f Train acc: %.2f TA %.2f ASR %.2f'%(epoch, train_loss, train_acc, ta, asr),
                args.checkpoint, 'train_log.txt')
            
            state ={
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
        
        elif args.atk == 'wanet':
            noise_grid, identity_grid = prepare(args)
            train_wanet(args, model, trainloader, criterion, optimizer, noise_grid, identity_grid, epoch)
            v_ta, v_asr, v_acc_cross, _ = test_wanet(args, model, valloader, noise_grid, identity_grid)
            ta, asr, acc_cross, _ = test_wanet(args, model, testloader, noise_grid, identity_grid)

            to_log_file('Clean acc %.2f BD acc %.2f Cross acc %.2f '%(v_ta, v_asr, v_acc_cross),
                args.checkpoint, 'train_log.txt')
            to_log_file('Clean acc %.2f BD acc %.2f Cross acc %.2f '%(ta, asr, acc_cross),
                args.checkpoint, 'train_log.txt')
            
            state = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    "identity_grid": identity_grid,
                    "noise_grid": noise_grid,
                }
        try:
            torch.save(state, os.path.join(args.checkpoint,'latest_ckpt.pth'), _use_new_zipfile_serialization=False)
        except:
            torch.save(state, os.path.join(args.checkpoint,'latest_ckpt.pth'))

        if not args.shuffle:
            if v_ta > best_ta or (v_ta > best_ta - 0.1 and v_asr > best_asr) :
                to_log_file('Saving Best ckpt...', args.checkpoint, 'train_log.txt')
                best_ta, best_asr = v_ta, v_asr
                try:
                    torch.save(state, os.path.join(args.checkpoint,'best_ckpt.pth'), _use_new_zipfile_serialization=False)
                except:
                    torch.save(state, os.path.join(args.checkpoint,'best_ckpt.pth'))
        else:
            if v_asr-v_ta > best_diff:
                to_log_file('Saving Best ckpt...', args.checkpoint, 'train_log.txt')
                best_diff = v_asr - v_ta
                try:
                    torch.save(state, os.path.join(args.checkpoint,'best_ckpt.pth'), _use_new_zipfile_serialization=False)
                except:
                    torch.save(state, os.path.join(args.checkpoint,'best_ckpt.pth'))
        ta_all.append(ta)
        v_ta_all.append(v_ta)
        plt.plot(ta_all, label='TA')
        plt.plot(v_ta_all, label='V_TA')
        
        asr_all.append(asr)
        v_asr_all.append(v_asr)
        plt.plot(asr_all, label='ASR')
        plt.plot(v_asr_all, label='V_ASR')

        plt.legend()
        plt.show()
        plt.savefig(os.path.join(args.checkpoint,'pic.png'))
        plt.close()
    
    if args.resume and args.epochs == 1:
        try:
            torch.save(state, os.path.join(args.checkpoint,'ckpt_epoch1.pth'), _use_new_zipfile_serialization=False)
        except:
            torch.save(state, os.path.join(args.checkpoint,'ckpt_epoch1.pth'))

def load_ckpt(path, model):
    params = torch.load(path)['state_dict']
    model_state = model.state_dict()
    new_ckpt = {}
    for k, v in params.items():
        if k in model_state:
            new_ckpt[k] = v
        else:
            print('%s not in'%k)

    model_state.update(new_ckpt)
    model.load_state_dict(model_state, strict=True)
    return model

def load_shuffle_ckpt(path, model):
    params = torch.load(path)['state_dict']
    model_state = model.state_dict()
    new_ckpt = {}
    for k, v in params.items():
        if k in model_state:
            if 'conv' in k:
                _, channels, _, _ = v.size()

                idx = torch.randperm(channels)
                v = v[:,idx,...]
            new_ckpt[k] = v
        else:
            print('%s not in'%k)
    model_state.update(new_ckpt)
    model.load_state_dict(model_state, strict=True)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Poisoning Benchmark")

    parser.add_argument('--atk', default='patch', choices=['patch', 'cla', 'wanet'])
    parser.add_argument('--rate',default=0.1,type=float)
    parser.add_argument('--target', default=0, type=int)
    parser.add_argument('--source', default=None, type=int)

    # for patch based atk
    parser.add_argument('--patch_size', default=5, type=int)
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--random_loc',action='store_true')
    parser.add_argument('--bottom_left',action='store_true')
    parser.add_argument('--multiple_targets', action='store_true')
    parser.add_argument('--target_num', default=2, type=int)

    # for wanet
    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--cross_ratio", type=float, default=2)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--s", type=float, default=0.8)
    parser.add_argument("--grid-rescale", type=float, default=1) 

    # std training params
    parser.add_argument("--seed", default=0, type=int, help="seed for seeding random processes.")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--lr_schedule", nargs="+", default=[100, 150], type=int)
    parser.add_argument("--lr_factor", default=0.1, type=float)
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs for training")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for training and testing")
    parser.add_argument("--optimizer", default="SGD", type=str, help="optimizer")

    # model and dataset
    parser.add_argument("--dataset", default="cifar10", choices=['cifar10','GTSRB'], type=str, help="dataset")
    parser.add_argument("--model", default="ResNet20s", choices=['ResNet20s', 'PreActResNet18','alexnet'], type=str)
    parser.add_argument("--no_normalize", action='store_true')
    
    # paths
    parser.add_argument("--data", default="")
    parser.add_argument("--checkpoint_root", default="")
    parser.add_argument("--resume", default="")
    
    # for shuffle
    parser.add_argument("--shuffle", action='store_true')

    # for randomized smoothing
    parser.add_argument("--test_resume", action='store_true')
    parser.add_argument("--shuffle_load", action='store_true')

    # for clean label attack
    parser.add_argument('--robust_model',default='')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)