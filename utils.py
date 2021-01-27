import torch
import torchvision
import torchvision.transforms
import matplotlib.pyplot as plt
import numpy as np

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def get_transform(phase):
        transforms = []
    if phase == "train":
        transforms.extend(
            [
                ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
            ]
        )
        
    transforms.extend(
        [
            Resize(512, 512),
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
        
    trans = Compose(transforms)
    return trans


def get_train_data_loader():
    transform = get_transform("train")
    trainset = torchvision.datasets.ImageFolder(root='./input/train_png', transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
