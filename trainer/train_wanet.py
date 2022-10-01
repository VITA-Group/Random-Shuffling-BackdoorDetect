from os import execv
import sys
sys.path.append("..")
from utils import to_log_file, AverageMeter

from trainer.wanet.utils import *
import torch.nn.functional as F

def train_wanet(args, model, train_loader, criterion, optimizer, noise_grid, identity_grid, epoch):

    model.train()

    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_cross = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0

    transforms = PostTensorTransform(args).to(args.device)
    
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

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

        total_preds, _ = model(total_inputs)
        loss = criterion(total_preds, total_targets)
        loss.backward()
        optimizer.step()

        total_sample += bs
        total_loss_ce += loss.detach() * bs

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
        print('Do not have any backdoor images')
        avg_acc_bd = 0
    try:
        avg_acc_cross = total_cross_correct * 100.0 / total_cross
    except:
        print('Do not have any cross images')
        avg_acc_cross = 0

    avg_loss_ce = total_loss_ce / total_sample

    to_log_file('Epoch %3d: Loss %.3f Clean acc %.2f BD acc %.2f Cross acc %.2f '%(epoch, avg_loss_ce, avg_acc_clean, avg_acc_bd, avg_acc_cross),
            args.checkpoint, 'train_log.txt')

def test_wanet(args, model, test_loader, noise_grid, identity_grid):
    
    model.eval()
    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0

    for batch_idx, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            x, y = x.to(args.device), y.to(args.device)
            bs = x.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean, _ = model(x)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == y)

            # Evaluate Backdoor
            grid_temps = (identity_grid + args.s * noise_grid / args.input_height) * args.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, args.input_height, args.input_height, 2).to(args.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / args.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = F.grid_sample(x, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            targets_bd = torch.ones_like(y) * args.target
            
            preds_bd, _ = model(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            inputs_cross = F.grid_sample(x, grid_temps2, align_corners=True)
            preds_cross, _ = model(inputs_cross)
            total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == y)

            acc_cross = total_cross_correct * 100.0 / total_sample

    return acc_clean.item(), acc_bd.item(), acc_cross.item(), None



def prepare(args):
    ins = torch.rand(1, 2, args.k, args.k) * 2 -1   
    ins = ins / torch.mean(torch.abs(ins))  # shape [1, 2, args.k, args.k] range [-1, 1]
    noise_grid = (
            F.upsample(ins, size=args.input_height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1).to(args.device)
        )
    array1d = torch.linspace(-1, 1, steps=args.input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].to(args.device)

    return noise_grid, identity_grid