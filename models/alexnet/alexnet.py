import torch.nn as nn
try:
    from advertorch.utils import NormalizeByChannelMeanStd
except:
    pass

'''
modified to fit dataset size
'''
NUM_CLASSES = 10


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, contain_normalize=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.contain_normalize = contain_normalize
        if contain_normalize:
            print('The normalize layer is contained in the network')
            self.normalize = NormalizeByChannelMeanStd(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    def forward(self, x):
        if self.contain_normalize:
            x = self.normalize(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        return self.classifier(x), x

def alexnet(num_classes, contain_normalize=False):
    return AlexNet(num_classes, contain_normalize=contain_normalize)