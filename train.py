import os
import cv2
import math
import time
import pickle
import random
import argparse
import albumentations
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import OneCycleLR
from torch.backends import cudnn


from dataset import LandmarkDataset, get_df, get_transforms
from metrics import global_average_precision_score
from model import ArcFace, apply_arcface_margin, GeM, DelgGlobal
from resnet import ResNet

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


def train_epoch(model, loader, optimizer, criterion):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        if not args.use_amp:
            logits_m = model(data)
            loss = criterion(logits_m, target)
            loss.backward()
            optimizer.step()
        else:
            logits_m = model(data)
            loss = criterion(logits_m, target)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
            
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    return train_loss

def val_epoch(model, valid_loader, criterion, get_output=False):

    model.eval()
    val_loss = []
    PRODS_M = []
    PREDS_M = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(valid_loader):
            data, target = data.cuda(), target.cuda()

            logits_m = model(data)

            lmax_m = logits_m.max(1)
            probs_m = lmax_m.values
            preds_m = lmax_m.indices

            PRODS_M.append(probs_m.detach().cpu())
            PREDS_M.append(preds_m.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits_m, target)
            val_loss.append(loss.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        PRODS_M = torch.cat(PRODS_M).numpy()
        PREDS_M = torch.cat(PREDS_M).numpy()
        TARGETS = torch.cat(TARGETS)

    if get_output:
        return LOGITS_M
    else:
        acc_m = (PREDS_M == TARGETS.numpy()).mean() * 100.
        y_true = {idx: target if target >=0 else None for idx, target in enumerate(TARGETS)}
        y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_M, PRODS_M))}
        gap_m = global_average_precision_score(y_true, y_pred_m)
        return val_loss, acc_m, gap_m

def optimizers_func(str):
    optimizer = 0
    if str == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr, dec)
    if str == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.init_lr)
    return optimizer

def main():


    # get augmentations
    transforms_train, transforms_val = get_transforms(args.image_size)

    # get train and valid dataset
    df_train = df[df['fold'] != args.fold]
    df_valid = df[df['fold'] == args.fold].reset_index(drop=True).query("index % 15==0")

    dataset_train = LandmarkDataset(df_train, 'train', 'train', transform=transforms_train)
    dataset_valid = LandmarkDataset(df_valid, 'train', 'val', transform=transforms_val)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    # model
    model = ModelClass(model)  # not sure
    model = model.cuda()

    # loss func
    def criterion(logits_m, target):
        loss_m = ArcFace(logits_m, target) # not sure
        return loss_m

    # optimizer
    optimizer = optimizers_func('adam')

    # # lr scheduler
    optimizer_anneal = torch.optim.lr_scheduler.OneCycleLR(optimizer,anneal_strategy='linear', max_lr=0.01, epochs = args.n_epochs)

    # train & valid loop
    gap_m_max = 0.
    model_file = os.path.join(args.model_dir, f'{args.enet_type}_fold{args.fold}.pth')
    for epoch in range(args.start_from_epoch, args.n_epochs+1):

        print(time.ctime(), 'Epoch:', epoch)

        train_sampler = torch.utils.data(dataset_train)
        train_sampler.set_epoch(epoch)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=train_sampler is None, sampler=train_sampler, drop_last=True)        

        train_loss = train_epoch(model, train_loader, optimizer_anneal, criterion)
        val_loss, acc_m, gap_m = val_epoch(model, valid_loader, criterion)

        if args.local_rank == 0:
            content = time.ctime() + ' ' + f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer_anneal.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, valid loss: {(val_loss):.5f}, acc_m: {(acc_m):.6f}, gap_m: {(gap_m):.6f}.'
            print(content)
            with open(os.path.join(args.log_dir, f'{args.enet_type}.txt'), 'a') as appender:
                appender.write(content + '\n')

            print('gap_m_max ({:.6f} --> {:.6f}). Saving model ...'.format(gap_m_max, gap_m))
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer_anneal.state_dict(),
                        }, model_file)            
            gap_m_max = gap_m

        if epoch == args.stop_at_epoch:
            print(time.ctime(), 'Training Finished!')
            break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_anneal.state_dict(),
    }, os.path.join(args.model_dir, f'{args.enet_type}_fold{args.fold}_final.pth'))


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnet':
        ModelClass = ResNet
    else:
      print("Error model not read")

    set_seed(0)
    main()
