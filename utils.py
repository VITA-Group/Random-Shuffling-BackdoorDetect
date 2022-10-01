import os
import sys
import torch
import torchvision

def more_config(args, print_log=True):
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.checkpoint = args.checkpoint_root + '/%s_%s'%(args.dataset, args.atk)
    
    if args.atk == 'patch':
        if args.random_loc and args.bottom_left:
            print('Specify the location')
            sys.exit()
        elif not args.random_loc and not args.bottom_left:
            args.upper_right = True
        else:
            args.upper_right = False
        
        args.checkpoint += '_rate_%.2f_%s'%(args.rate, 'color' if args.color else 'black')
    
    elif args.atk == 'cla':
        if args.random_loc and args.bottom_left:
            print('Specify the location')
            sys.exit()
        elif not args.random_loc and not args.bottom_left:
            args.upper_right = True
        else:
            args.upper_right = False
        
        args.checkpoint += '_rate_%.2f_%s'%(args.rate, 'cla')
    
    elif args.atk == 'wanet':
        args.checkpoint += '_rate_%.2f_k_%d_s_%.1f'%(args.rate, args.k, args.s)
    
    if args.shuffle:
        args.checkpoint = os.path.join(
            args.checkpoint,
            args.model+'_shuffle_seed{0}'.format(args.seed)
            )
    else:
        args.checkpoint = os.path.join(
            args.checkpoint,
            args.model+'_original_seed{0}'.format(args.seed)
            )
    
    train_log = "train_log.txt"
    for arg in vars(args):
        to_log_file(arg+' {0}'.format(getattr(args, arg)), args.checkpoint, train_log, print_log)

    if args.dataset.lower() == 'cifar10':
        args.num_classes = 10
        args.input_height, args.input_width, args.input_channel = 32, 32, 3
    elif args.dataset.lower() == 'gtsrb':
        args.num_classes = 43
        args.input_height, args.input_width, args.input_channel = 32, 32, 3

def to_log_file(out_dict, out_dir, log_name="output_default.txt", print_log=True):
    """Function to write the logfiles
    input:
        out_dict:   Dictionary of content to be logged
        out_dir:    Path to store the output_default file
        log_name:   Name of the output_default file
    return:
        void
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)

    with open(fname, "a") as f:
        f.write(str(out_dict) + "\n")
    if print_log:
        print(str(out_dict))

def adjust_learning_rate(optimizer, epoch, lr_schedule, lr_factor):
    """Function to decay the learning rate
    input:
        optimizer:      Pytorch optimizer object
        epoch:          Current epoch number
        lr_schedule:    Learning rate decay schedule list
        lr_factor:      Learning rate decay factor
    return:
        void
    """
    if epoch in lr_schedule:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_factor
        print(
            "Adjusting learning rate ",
            param_group["lr"] / lr_factor,
            "->",
            param_group["lr"],
        )
    return

def show_images(img, path):
    torchvision.utils.save_image(img, path, normalize=True)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count