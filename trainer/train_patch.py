import torch
import time
from utils import AverageMeter

def train_patch(model, trainloader, optimizer, criterion, device, shuffle=False):
    
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    #entropy_meter = AverageMeter()

    #unloader = transforms.ToPILImage()
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        bs = inputs.shape[0]

        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        equal = predicted.eq(targets)
        correct += equal.sum().item()
        
        # calculate entropy on clean samples
        # count = torch.bincount(predicted)
        # count = count[torch.nonzero(count)]
        # freq = count / bs

        # entropy = -torch.sum(freq * torch.log2(freq))
        # entropy_meter.update(entropy.item(), bs)

        if shuffle:
            model = shuffle_ckpt(model)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    return train_loss, acc, None

def test_patch(model, testloader, device, shuffle=False):

    model.eval()
    natural_correct = 0
    total = 0

    #entropy_meter = AverageMeter()

    #to_tar = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            # if batch_idx > 80:
            #     break

            bs = inputs.shape[0]

            inputs, targets = inputs.to(device), targets.to(device)
            _outputs, _ = model(inputs)
            _, natural_predicted = _outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()

            #to_tar += natural_predicted.eq(0).sum().item()

            total += targets.size(0)

            # calculate entropy on clean samples
            # count = torch.bincount(natural_predicted)
            # count = count[torch.nonzero(count)]
            # freq = count / bs

            # entropy = -torch.sum(freq * torch.log2(freq))
            # entropy_meter.update(entropy.item(), bs)

            if shuffle:
                model = shuffle_ckpt(model)
    
    natural_acc = 100.0 * natural_correct / total

    return natural_acc, None

def shuffle_ckpt(model):
    model_state = model.state_dict()
    new_ckpt = {}
    for k, v in model_state.items():
        if 'conv' in k:
            _, channels, _, _ = v.size()

            idx = torch.randperm(channels)
            v = v[:,idx,...]

        new_ckpt[k] = v
    model_state.update(new_ckpt)
    model.load_state_dict(model_state, strict=True)
    return model