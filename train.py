import os
import cv2
import math
import time
import datetime
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from logger import setup_logger
from configs import get_cfg_defaults
from dataset import LandmarkDataset, get_df, get_transforms
from metrics import global_average_precision_score_val
from model import DelgGlobal

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--train-step', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--init-lr', type=float, default=1e-4)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--start-from-epoch', type=int, default=1)
    parser.add_argument('--stop-at-epoch', type=int, default=10)
    parser.add_argument('--use-amp', action='store_false')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--fold', type=int, default=0)
    args, _ = parser.parse_known_args()
    return args


def train_epoch(cfg, model, loader, optimizer, criterion):
    device = cfg.SYSTEM.DEVICE
    model = model.train()
    train_loss = []
    bar = tqdm(loader)
    for i, (data, target) in enumerate(bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        _, logits_m = model(data, target)
        loss = criterion(logits_m, target)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    return train_loss

def val_epoch(cfg, model, df_gallery, valid_loader, gallery_loader, criterion, get_output=False):
    device = cfg.SYSTEM.DEVICE
    model = model.eval()
    val_loss = []
    PRODS_M = []
    PREDS_M = []
    TARGETS = []
    
    PROBS = []
    PREDS = []
    TOP_K = 5
    with torch.no_grad():
        """
        Compute global descriptor for train set
        """
        feats = []
        gallery_landmark_id = []
        for (data, target) in tqdm(gallery_loader): 
            data, target = data.to(device), target.to(device)
            feat, _ = model(data, labels=None, training=False)
            feats.append(feat.detach().cpu())
            gallery_landmark_id.append(target.detach().cpu())
            
        gallery_landmark_id = torch.cat(gallery_landmark_id).numpy()
        feats = torch.cat(feats)
        feats = feats.to(device)
        
        """
        Compute global descriptor for validation set
        """
        for (data, target) in tqdm(valid_loader):
            data, target = data.to(device), target.to(device)

            feat, logits_m = model(data, target, training=False)
            #softmax_prob = F.softmax(logits_m)
            
            # cosine similarity of the current validation image to the images in the train set
            similarity = feat.mm(feats.t())
            
            # Retrieve top-k images from the train set
            (values, indices) = torch.topk(similarity, TOP_K, dim=1)
            probs = values
            preds = indices # indices in the train set
            PROBS.append(probs.detach().cpu())
            PREDS.append(preds.detach().cpu())
            
            #lmax_m = logits_m.max(1)
            #probs_m = lmax_m.values
            #preds_m = lmax_m.indices

            #PRODS_M.append(probs_m.detach().cpu())
            #PREDS_M.append(preds_m.detach().cpu())
            
            TARGETS.append(target.detach().cpu())
            
            loss = criterion(logits_m, target)
            val_loss.append(loss.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        #PRODS_M = torch.cat(PRODS_M).numpy()
        #PREDS_M = torch.cat(PREDS_M).numpy()
        
        PROBS = torch.cat(PROBS).numpy()
        PREDS = torch.cat(PREDS).numpy()
        TARGETS = torch.cat(TARGETS)
        
        # Convert the train set indices to landmark ids
        PREDS = gallery_landmark_id[PREDS]
        
        """
        Get the final prediction of the landmark with the highest score from the top-k retrieved images
        """
        PROBS_F = []
        PREDS_F = []
        for i in tqdm(range(PREDS.shape[0])):
            tmp = {}
            for k in range(TOP_K):
                pred_k = PREDS[i, k]
                tmp[pred_k] = tmp.get(pred_k, 0.) + float(PROBS[i, k])
            
            pred, conf = max(tmp.items(), key=lambda x: x[1])
            PREDS_F.append(pred)
            PROBS_F.append(conf)
            
            
    if get_output:
        return logits_m
    else:
        acc_m = (PREDS_F == TARGETS.numpy()).mean() * 100.
        y_true = {idx: target if target >=0 else None for idx, target in enumerate(TARGETS)}
        pred_f = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_F, PROBS_F))}
        #y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_M, PRODS_M))}
        gap_m = global_average_precision_score_val(y_true, pred_f)
        return val_loss, acc_m, gap_m

def optimizers_func(cfg, model):
    #optimizer = 0
    if cfg.TRAIN.OPTIM.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(),
                              lr=cfg.TRAIN.OPTIM.BASE_LR,
                              betas=cfg.TRAIN.OPTIM.ADAM_BETAS)
    if cfg.TRAIN.OPTIM.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.TRAIN.OPTIM.BASE_LR,
                              momentum=cfg.TRAIN.OPTIM.SGD_MOMENTUM)
    return optimizer

def main():
    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/delg.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "options",
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = cfg.SYSTEM.DEVICE
    
    # get dataframe
    df, num_classes = get_df(cfg.DATASET.TRAIN_DATA_DIR, cfg.DATASET.TRAIN_CSV)
    #print("number of classes: %6d" % num_classes)

    # get augmentations
    transforms_train, transforms_val = get_transforms(cfg.DATASET.IMAGE_SIZE)
    
    # get dataset
    dataset_train = LandmarkDataset(df, 'train', transform=transforms_train)
    dataset_valid = LandmarkDataset(df, 'val', transform=transforms_val)
    
    dataset_train, _ = torch.utils.data.random_split(dataset_train, 
                                                     [math.ceil(0.8*len(dataset_train)), math.floor(0.2*len(dataset_train))],
                                                     generator=torch.Generator().manual_seed(0))
    _, dataset_valid = torch.utils.data.random_split(dataset_valid,
                                                     [math.ceil(0.8*len(dataset_valid)), math.floor(0.2*len(dataset_valid))],
                                                     generator=torch.Generator().manual_seed(0))
    valid_loader = torch.utils.data.DataLoader(dataset_valid, 
                                               batch_size=cfg.TRAIN.BATCH_SIZE, 
                                               num_workers=cfg.SYSTEM.NUM_WORKERS)
    
    df_gallery, _ = get_df(cfg.DATASET.TRAIN_DATA_DIR, cfg.DATASET.TRAIN_CSV)
    dataset_gallery = LandmarkDataset(df_gallery, 'val', transform=transforms_val)
    dataset_gallery, _ = torch.utils.data.random_split(dataset_gallery, 
                                                     [math.ceil(0.8*len(dataset_gallery)), math.floor(0.2*len(dataset_gallery))],
                                                     generator=torch.Generator().manual_seed(0))
    gallery_loader = torch.utils.data.DataLoader(dataset_gallery, 
                                                 batch_size=cfg.TRAIN.BATCH_SIZE, 
                                                 num_workers=cfg.SYSTEM.NUM_WORKERS)
    
    # model
    if cfg.MODEL.MODEL == 'delg_global':
        model = DelgGlobal(num_classes, embedding_size=1024, pretrained=True)  
    model = model.to(device)
    #model = DDP(model, device_ids=[0])
    
    if cfg.TRAIN.FREEZE_BACKBONE:
        for param in model.backbone_to_conv4.parameters():
            param.requires_grad = False
        
        for param in model.backbone_conv5.parameters():
            param.requires_grad = False
        
    # loss func
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optimizers_func(cfg, model)

    # logging
    ts = time.time()
    logger = setup_logger('training_info', 'logger/train', file_name='train-log-' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S') + '.txt')
    logger.info(cfg)
    logger.info('numer of classes: %d' % num_classes)
    
    # train & valid loop
    gap_m_max = 0.
    model_file = os.path.join(cfg.MODEL.MODEL_DIR, cfg.MODEL.BACKBONE + '_' + cfg.MODEL.MODEL + '.pth')
    for epoch in range(cfg.TRAIN.EPOCHS):
        
        if epoch % cfg.TRAIN.OPTIM.LR_DECAY_FREQUENCY == 0 and epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= cfg.TRAIN.OPTIM.LR_DECAY_RATE    
                    
        
        print(time.ctime(), 'Epoch:', epoch)
        tic = time.time()

        train_loader = torch.utils.data.DataLoader(dataset_train, 
                                                   batch_size=cfg.TRAIN.BATCH_SIZE, 
                                                   num_workers=cfg.SYSTEM.NUM_WORKERS, 
                                                   shuffle=True)

        train_loss = train_epoch(cfg, model, train_loader, optimizer, criterion)
        val_loss, acc_m, gap_m = val_epoch(cfg, model, df_gallery, valid_loader, gallery_loader, criterion)
        
        toc = time.time()
        content = 'epoch: %d, time: %.2f s, lr: %.5f, train loss: %.5f, validation loss: %.5f, acc_m: %.5f, gap_m: %.5f' % (epoch, toc - tic, optimizer.param_groups[0]['lr'], np.mean(train_loss), val_loss, acc_m, gap_m)
        logger.info(content)
        if gap_m > gap_m_max:
            print('saving best model')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, model_file)  
            gap_m_max = gap_m
        


if __name__ == '__main__':
    main()
