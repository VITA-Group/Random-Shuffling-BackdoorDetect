import torch
import random
# import kornia.augmentation as A
from torchvision import transforms

# class ProbTransform(torch.nn.Module):
#     def __init__(self, f, p=1):
#         super(ProbTransform, self).__init__()
#         self.f = f
#         self.p = p

#     def forward(self, x):  # , **kwargs):
#         if random.random() < self.p:
#             return self.f(x)
#         else:
#             return x

# class PostTensorTransform(torch.nn.Module):
#     def __init__(self, args):
#         super(PostTensorTransform, self).__init__()
#         self.random_crop = ProbTransform(
#             A.RandomCrop((args.input_height, args.input_width), padding=args.random_crop), p=0.8
#         )
#         self.random_rotation = ProbTransform(A.RandomRotation(args.random_rotation), p=0.5)
#         if args.dataset == "cifar10":
#             self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

#     def forward(self, x):
#         for module in self.children():
#             x = module(x)
#         return x

class PostTensorTransform(torch.nn.Module):
    def __init__(self, args):
        super(PostTensorTransform, self).__init__()

        self.transform_list = [transforms.ToPILImage()]
        if random.random() < 0.8:
            self.transform_list.append(transforms.RandomCrop(args.input_height, padding=args.random_crop))
        if random.random() < 0.5:
            self.transform_list.append(transforms.RandomRotation(degrees=args.random_rotation))
        if args.dataset == 'cifar10' and random.random() < 0.5:
            self.transform_list.append(transforms.RandomHorizontalFlip())
        self.transform_list.append(transforms.ToTensor())
        # if args.dataset == 'cifar10':
        #     self.transform_list.append(transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)))
        self.transform = transforms.Compose(self.transform_list)
        
    def forward(self, x):
        return torch.stack([self.transform(x[i]) for i in range(x.shape[0])], dim=0)
