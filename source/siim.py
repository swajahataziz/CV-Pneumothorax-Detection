from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor


class SimDataset(Dataset):
    def __init__(self, input_dir, image_data, size, mean, std, phase):
        self.input_dir = input_dir
        self.image_data = image_data

        self.transform = get_transforms(phase, size, mean, std)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        
        encoded_pix = self.image_data.iloc[idx, 1]
        image_id = self.image_data.iloc[idx, 0]
        image_path = self.input_dir +'/'+ str(image_id) +".png"
        image = cv2.imread(image_path)

        mask = np.zeros([1024, 1024])
        
        if encoded_pix != '-1':
            mask += decode(encoded_pix)
            
        if self.transform:
            augmented = self.transform(image=image , mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        return [image, mask]    

def load_data(
    fold,
    total_folds,
    data_folder,
    df_path,
    phase,
    size,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
):
    df_all = pd.read_csv(df_path)
    df = df_all.drop_duplicates('ImageId')
    df_with_mask = df[df[" EncodedPixels"] != "-1"]
    df_with_mask['has_mask'] = 1
    df_without_mask = df[df[" EncodedPixels"] == "-1"]
    df_without_mask['has_mask'] = 0
    df_without_mask_sampled = df_without_mask.sample(len(df_with_mask), random_state=69) # random state is imp
    df = pd.concat([df_with_mask, df_without_mask_sampled])
    
    #NOTE: equal number of positive and negative cases are chosen.
    
    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    
    for train_index, val_index in kfold.split(df["ImageId"], df["has_mask"]):
        print("TRAIN:", train_index, "TEST:", val_index)
        train_df, val_df = df.iloc[train_index], df.iloc[val_index]

    df = train_df if phase == "train" else val_df
    # NOTE: total_folds=5 -> train/val : 80%/20%
        
    image_dataset = SimDataset(data_folder, df, size, mean, std, phase)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader

def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
#                 HorizontalFlip(),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=10, # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
#                 GaussNoise(),
            ]
        )
    list_transforms.extend(
        [
            Resize(size, size),
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )

    list_trfms = Compose(list_transforms)
    return list_trfms

def decode(rle, height=1024, width=1024, fill_value=1):
    decoded = np.zeros((height, width), np.float32)
    decoded = decoded.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        decoded[start: end] = fill_value
        start = end
    decoded = decoded.reshape(width, height).T
    return decoded

def encode(data):
    data = data.T.flatten()
    start = np.where(data[1:] > data[:-1])[0]+1
    end = np.where(data[:-1] > data[1:])[0]+1
    length = end-start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i]-end[i-1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle