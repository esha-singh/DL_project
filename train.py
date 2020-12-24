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
from model import Delg

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


def train_global_epoch(cfg, model, loader, criterion, optimizer):
    for param in model.backbone_to_conv4.parameters():
        param.requires_grad = True
    for param in model.backbone_conv5.parameters():
        param.requires_grad = True
        
    device = cfg.SYSTEM.DEVICE
    model = model.train()
    global_train_loss = []
    attn_train_loss = []
    bar = tqdm(loader)
    for i, (data, target) in enumerate(bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        global_f, global_logits = model(data, target, global_only=True, training=True)
        loss_global = criterion(global_logits, target)
        loss_global.backward() 
        optimizer.step()
            
        torch.cuda.synchronize()
        
        loss_global_np = loss_global.detach().cpu().numpy()
        global_train_loss.append(loss_global_np)
        global_smooth_loss = sum(global_train_loss[-100:]) / min(len(global_train_loss), 100)
        bar.set_description('global loss: %.5f, global smth: %.5f' % (loss_global_np, global_smooth_loss))
    
    return global_train_loss
    

def train_local_epoch(cfg, model, loader, criterion, optimizer):
    for param in model.backbone_to_conv4.parameters():
        param.requires_grad = False
    for param in model.backbone_conv5.parameters():
        param.requires_grad = False
            
    device = cfg.SYSTEM.DEVICE
    model = model.train()
    attn_train_loss = []
    bar = tqdm(loader)
    for i, (data, target) in enumerate(bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        global_f, global_logits, attn_f, attn_logits, score, prob = model(data, target, global_only=False, training=True)
        loss_attn = criterion(attn_logits, target)
        loss_attn.backward()
        optimizer.step()
            
        torch.cuda.synchronize()
        
        loss_attn_np = loss_attn.detach().cpu().numpy()
        attn_train_loss.append(loss_attn_np)
        attn_smooth_loss = sum(attn_train_loss[-100:]) / min(len(attn_train_loss), 100)
        bar.set_description('attn loss: %.5f, attn smth: %.5f' % (loss_attn_np, attn_smooth_loss))
       
    return attn_train_loss

    
def val_epoch(cfg, model, df_gallery, valid_loader, gallery_loader, criterion, global_only=False):
    device = cfg.SYSTEM.DEVICE
    model = model.eval()
    global_val_loss = []
    attn_val_loss = []
    GLOBAL_PROBS = []
    GLOBAL_PREDS = []
    ATTN_PROBS = []
    ATTN_PREDS = []
    TARGETS = []
    
    PROBS = []
    PREDS = []
    TOP_K = 5
    with torch.no_grad():
        """
        #Compute global descriptor for train set
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
        if not global_only:
            for (data, target) in tqdm(valid_loader):
                data, target = data.to(device), target.to(device)
            
                global_f, global_logits, attn_f, attn_logits, score, prob = model(data, target, training=False)
                #global_softmax_prob = F.softmax(global_logits, dim=1)
                #attn_softmax_prob = F.softmax(attn_logits, dim=1)
                
                global_max = global_logits.max(1)
                global_probs = global_max.values
                global_preds = global_max.indices
                
                attn_max = attn_logits.max(1)
                attn_probs = attn_max.values
                attn_preds = attn_max.indices
                
                GLOBAL_PROBS.append(global_probs.detach().cpu())
                GLOBAL_PREDS.append(global_preds.detach().cpu())
                ATTN_PROBS.append(attn_probs.detach().cpu())
                ATTN_PREDS.append(attn_preds.detach().cpu())
                
                TARGETS.append(target.detach().cpu())
                
                global_loss = criterion(global_logits, target)
                global_val_loss.append(global_loss.detach().cpu().numpy())
                attn_loss = criterion(attn_logits, target)
                attn_val_loss.append(attn_loss.detach().cpu().numpy())
        
            global_val_loss = np.mean(global_val_loss)
            attn_val_loss = np.mean(attn_val_loss)
            GLOBAL_PROBS = torch.cat(GLOBAL_PROBS).numpy()
            GLOBAL_PREDS = torch.cat(GLOBAL_PREDS).numpy()
            ATTN_PROBS = torch.cat(ATTN_PROBS).numpy()
            ATTN_PREDS = torch.cat(ATTN_PREDS).numpy()
            
            #PROBS = torch.cat(PROBS).numpy()
            #PREDS = torch.cat(PREDS).numpy()
            TARGETS = torch.cat(TARGETS)
        else:
            #Compute global descriptor for validation set
            for (data, target) in tqdm(valid_loader):
                data, target = data.to(device), target.to(device)
                
                
                feat, global_logits = model(data, target, global_only=False, training=False)
                global_softmax_prob = F.softmax(global_logits)
                """
                # cosine similarity of the current validation image to the images in the train set
                similarity = feat.mm(feats.t())
                
                # Retrieve top-k images from the train set
                (values, indices) = torch.topk(similarity, TOP_K, dim=1)
                probs = values
                preds = indices # indices in the train set
                PROBS.append(probs.detach().cpu())
                PREDS.append(preds.detach().cpu())
                """
                global_max = global_softmax_prob.max(1)
                global_probs = global_max.values
                global_preds = global_max.indices
                
                GLOBAL_PROBS.append(global_probs.detach().cpu())
                GLOBAL_PREDS.append(global_preds.detach().cpu())
                #lmax_m = logits_m.max(1)
                #probs_m = lmax_m.values
                #preds_m = lmax_m.indices
    
                #PRODS_M.append(probs_m.detach().cpu())
                #PREDS_M.append(preds_m.detach().cpu())
                
                TARGETS.append(target.detach().cpu())
                
                global_loss = criterion(global_logits, target)
                global_val_loss.append(global_loss.detach().cpu().numpy())
    
            global_val_loss = np.mean(global_val_loss)
            GLOBAL_PROBS = torch.cat(GLOBAL_PROBS).numpy()
            GLOBAL_PREDS = torch.cat(GLOBAL_PREDS).numpy()
            #PRODS_M = torch.cat(PRODS_M).numpy()
            #PREDS_M = torch.cat(PREDS_M).numpy()
            
            #PROBS = torch.cat(PROBS).numpy()
            #PREDS = torch.cat(PREDS).numpy()
            TARGETS = torch.cat(TARGETS)
            
            # Convert the train set indices to landmark ids
            PREDS = gallery_landmark_id[PREDS]
        
        """
        #Get the final prediction of the landmark with the highest score from the top-k retrieved images
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
        """
        
    if not global_only:
        global_acc = (GLOBAL_PREDS == TARGETS.numpy()).mean() * 100.
        print(np.linalg.norm(ATTN_PREDS))
        print(ATTN_PREDS)
        print(GLOBAL_PREDS)
        print(np.linalg.norm(TARGETS))
        attn_acc = (ATTN_PREDS == TARGETS.numpy()).mean() * 100.
        y_true = {idx: target if target >=0 else None for idx, target in enumerate(TARGETS)}
        #pred_f = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_F, PROBS_F))}
        y_pred_global = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(GLOBAL_PREDS, GLOBAL_PROBS))}
        y_pred_attn = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(ATTN_PREDS, ATTN_PROBS))}
        global_gap = global_average_precision_score_val(y_true, y_pred_global)
        attn_gap = global_average_precision_score_val(y_true, y_pred_attn)
        
        return global_val_loss, attn_val_loss, global_acc, attn_acc, global_gap, attn_gap
    
    else:
        global_acc = (GLOBAL_PREDS == TARGETS.numpy()).mean() * 100.
        y_true = {idx: target if target >=0 else None for idx, target in enumerate(TARGETS)}
        y_pred_global = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(GLOBAL_PREDS, GLOBAL_PROBS))}
        global_gap = global_average_precision_score_val(y_true, y_pred_global)
        
        return global_val_loss, global_acc, global_gap

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
    parser.add_argument("--config-file", default="configs/delg.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--global-only", action='store_true', help="only train global part")
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
    model = Delg(num_classes, embedding_size=1024, pretrained=True)  
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
    #attn_optimizer = optimizers_func(cfg, model.attention)

    # logging
    ts = time.time()
    logger = setup_logger('training_info', 'logger/train', file_name='train-log-' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S') + '.txt')
    logger.info(cfg)
    logger.info('numer of classes: %d' % num_classes)
    
    # train & valid loop
    #gap_m_max = 0.
    model_file = os.path.join(cfg.MODEL.MODEL_DIR, cfg.MODEL.BACKBONE + '_' + cfg.MODEL.MODEL + '.pth')
    for epoch in range(cfg.TRAIN.EPOCHS):
        
        if epoch % cfg.TRAIN.OPTIM.LR_DECAY_FREQUENCY == 0 and epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= cfg.TRAIN.OPTIM.LR_DECAY_RATE    
                    
        
        print(time.ctime(), 'Epoch:', epoch)

        train_loader = torch.utils.data.DataLoader(dataset_train, 
                                                   batch_size=cfg.TRAIN.BATCH_SIZE, 
                                                   num_workers=cfg.SYSTEM.NUM_WORKERS, 
                                                   shuffle=True)
        
        global_train_loss = train_global_epoch(cfg, model, train_loader, criterion, optimizer)
        if not args.global_only:
            attn_train_loss = train_local_epoch(cfg, model, train_loader, criterion, optimizer)
        
        
        if not args.global_only:
            global_val_loss, attn_val_loss, global_acc, attn_acc, global_gap, attn_gap = val_epoch(cfg, model, df_gallery, valid_loader, gallery_loader, criterion, args.global_only)
            content = 'epoch: %d, lr: %.5f, global train loss: %.5f, local train loss: %.5f, global validation loss: %.5f, local validation loss: %.5f, global_acc: %.5f, local_acc: %.5f, global_gap: %.5f, local_gap: %.5f' % (epoch, 
                                                                                                                                                                                                                                 optimizer.param_groups[0]['lr'], 
                                                                                                                                                                                                                                 np.mean(global_train_loss), 
                                                                                                                                                                                                                                 np.mean(attn_train_loss), 
                                                                                                                                                                                                                                 global_val_loss, 
                                                                                                                                                                                                                                 attn_val_loss,
                                                                                                                                                                                                                                 global_acc, 
                                                                                                                                                                                                                                 attn_acc,
                                                                                                                                                                                                                                 global_gap,
                                                                                                                                                                                                                                 attn_gap)
        else:
            global_val_loss, global_acc, global_gap = val_epoch(cfg, model, df_gallery, valid_loader, gallery_loader, criterion, args.global_only)
            content = 'epoch: %d, lr: %.5f, global train loss: %.5f, global validation loss: %.5f, global_acc: %.5f, global_gap: %.5f' % (epoch, optimizer.param_groups[0]['lr'], np.mean(global_train_loss), global_val_loss, global_acc, global_gap)
        
        logger.info(content)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, model_file)  
        """
        if gap_m > gap_m_max:
            print('saving best model')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, model_file)  
            gap_m_max = gap_m
        """
        


if __name__ == '__main__':
    main()
