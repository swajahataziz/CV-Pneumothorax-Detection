import argparse
import json
import logging
import sagemaker_containers
import time

import os
import cv2
import pdb
import warnings
import random
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist

from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler

import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp
from loss import *

from siim import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ## torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
        self.criterion = MixedLoss(10.0, 2.0)
        
        if torch.cuda.device_count() > 1:
            logger.info("Gpu count: {}".format(torch.cuda.device_count()))
            self.net = nn.DataParallel(self.net)
        self.net = self.net.to(self.device)
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    
class Trainer(object):
    def __init__(self, args):
        self.fold = 1
        self.total_folds = 5
        self.num_workers = 6
        self.batch_size = {"train": args.batch_size, "val": args.batch_size}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = args.lr #5e-4
        self.num_epochs = args.epochs
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        ## self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ## torch.set_default_tensor_type("torch.cuda.FloatTensor")
        ## self.net = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
        ## self.criterion = MixedLoss(10.0, 2.0)
        
        # if torch.cuda.device_count() > 1:
        #    logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        #    self.net = nn.DataParallel(self.net)
        self.model = Unet()
        self.net = self.model.net
        
        self.optimizer = optimizer.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        cudnn.benchmark = True

        self.dataloaders = {
            phase: load_data(
                fold=1,
                total_folds=5,
                data_folder= args.data_dir,
                df_path=args.data_dir+'/train-rle-cleansed.csv',
                phase=phase,
                size=512,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
            
    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | time: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            loss, outputs = self.model.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss
    
    def train(self, args):
        is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
        logger.debug("Distributed training - {}".format(is_distributed))

        if is_distributed:
            # Initialize the distributed environment.
            world_size = len(args.hosts)
            os.environ['WORLD_SIZE'] = str(world_size)
            host_rank = args.hosts.index(args.current_host)
            os.environ['RANK'] = str(host_rank)
            dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
            logger.info(
                'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                    args.dist_backend,
                    dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                    dist.get_rank(), torch.cuda.is_available(), args.num_gpus))

        
        model = Unet()
        
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
        print('Finished Training')
        return _save_model(self.net, args.model_dir)
            
def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')
    parser.add_argument('--epochs', type=int, default=20, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='initial learning rate (default: 0.001)')

    env = sagemaker_containers.training_env()
    parser.add_argument('--hosts', type=json.loads, default=os.environ['SM_HOSTS'])
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    trainer = Trainer(parser.parse_args())
    trainer.train(parser.parse_args())
    

def model_fn(model_dir):
    logger.info('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    model.eval()
    
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model_prefix = 'net.'
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        #state_dict = torch.load(f)
        #from collections import OrderedDict
        #new_state_dict = OrderedDict()
        #for k, v in state_dict.items():
            #name = model_prefix+k # remove `module.`
            #new_state_dict[name] = v
        ## load params
        #model.load_state_dict(new_state_dict)
        
        model.load_state_dict(torch.load(f))
    return model.to(device)

