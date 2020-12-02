# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:40:53 2020

@author: tedkuo
"""
import os
import cv2
import glob
import math
import time
import datetime
import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm as tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from logger import setup_logger
from configs import get_cfg_defaults
from dataset import LandmarkDataset, get_df, get_transforms
from metrics import global_average_precision_score_test
from model import DelgGlobal


def test(cfg, model, df_gallery, df_query, df_sol, gallery_loader, query_loader, save_gallery_feats=False):
    device = cfg.SYSTEM.DEVICE
    
    TOP_K = 5
    #CLS_TOP_K = 5
    with torch.no_grad():
        """
        Compute global descriptor for gallery set
        """
        feats = []
        if save_gallery_feats:
            for img_0, img_1, img_2 in tqdm(gallery_loader): 
                img_0, img_1, img_2 = img_0.to(device), img_1.to(device), img_2.to(device)
                
                #feat_0, _ = model(img_0, training=False)
                feat, _ = model(img_1, training=False)
                #feat_2, _ = model(img_2, training=False)
                
                #feat = (feat_0 + feat_1 + feat_2)/3
                #feat = torch.cat([feat_0, feat_1, feat_2], dim=1)
                #feat = F.normalize(feat)
                #feat = torch.cat([feat_0, feat_1, feat_2], dim=1)
                
                feats.append(feat.detach().cpu())
            feats = torch.cat(feats)
            torch.save(feats, 'gallery_feats.pt')
              
        PROBS = []
        PREDS = []
        #PRODS_M = []
        #PREDS_M = []      
        feats = torch.load('gallery_feats.pt')
        print(feats.shape)
        feats = feats.to(device)
        
        """
        Compute global descriptor for query set
        """
        for img_0, img_1, img_2 in tqdm(query_loader):
            img_0, img_1, img_2 = img_0.to(device), img_1.to(device), img_2.to(device)
            
            #feat_0, logits_0 = model(img_0, training=False)
            feat, logits_1 = model(img_1, training=False)
            #feat_2, logits_2 = model(img_2, training=False)
            #feat = (feat_0 + feat_1 + feat_2)/3
            #feat = torch.cat([feat_0, feat_1, feat_2], dim=1)
            #feat = F.normalize(feat)
            
            # cosine similarity of the current query image to the images in the gallery set
            similarity = feat.mm(feats.t())
            
            # Retrieve top-k images from the train set
            (values, indices) = torch.topk(similarity, TOP_K, dim=1)
            probs = values
            preds = indices # indices in the gallery set
            PROBS.append(probs.detach().cpu())
            PREDS.append(preds.detach().cpu())

        PROBS = torch.cat(PROBS).numpy()
        PREDS = torch.cat(PREDS).numpy()
        
        # Convert the gallery set indices to landmark ids
        gallery_landmark_id = df_gallery['landmark_id'].values
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
            

        y_true = {idx: target for idx, target in enumerate(df_query['landmark_id'])}
        pred_f = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_F, PROBS_F))}
        gap_m, acc_m = global_average_precision_score_test(y_true, pred_f)
        
        return gap_m, acc_m
        
    
def main():
    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/delg.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--save-gallery-feats", action='store_true', help="save global features of gallery set")
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = cfg.SYSTEM.DEVICE
    
    # get dataframe
    df_gallery = get_df(cfg.DATASET.TRAIN_DATA_DIR, 'train.csv', 'test')
    df_query = get_df(cfg.DATASET.TEST_DATA_DIR, 'sample_submission.csv', 'test')
    df_sol = pd.read_csv('recognition_solution_v2.1.csv')
    
    # create labels for test csv 
    id2sol = {id_: list(map(int, sol.split())) if isinstance(sol, str) else None for id_, sol in zip(df_sol['id'], df_sol['landmarks'])} 
    df_query['landmark_id'] = df_query['id'].map(id2sol)

    #df_query = df_query.head(100)
    #df_gallery = df_gallery.head(50000)
    
    
    #Eliminate the landmarks that are not in the gallery (train) set
    arr = df_gallery['landmark_id'].values
    arr = list(dict.fromkeys(arr))
    drop = []
    for idx, row in df_query.iterrows():
        table = []
        if row['landmark_id'] is not None:
            for landmark in row['landmark_id']:
                table.append(landmark in arr)
            if not any(table):
                drop.append(idx)
    df_query.drop(df_query.index[drop], inplace=True)
    
    
    """
    # create csv for subset
    indexes_to_drop = []
    for index, row in tqdm(df_gallery.iterrows()):
        if not os.path.exists(row.filepath):
            indexes_to_drop.append(index)
    df_gallery.drop(df_gallery.index[indexes_to_drop], inplace=True)
    df_gallery.to_csv('train_sub.csv')
    """
    #print("number of classes: %6d" % num_classes)

    # get augmentations
    transforms_test = get_transforms(cfg.DATASET.IMAGE_SIZE, mode='test')
    
    # get dataset
    dataset_gallery = LandmarkDataset(df_gallery, 'test', transform=transforms_test)
    dataset_query = LandmarkDataset(df_query, 'test', transform=transforms_test)
    
    gallery_loader = torch.utils.data.DataLoader(dataset_gallery, 
                                                 batch_size=cfg.EVAL.BATCH_SIZE, 
                                                 num_workers=cfg.SYSTEM.NUM_WORKERS)
    query_loader = torch.utils.data.DataLoader(dataset_query, 
                                               batch_size=cfg.EVAL.BATCH_SIZE, 
                                               num_workers=cfg.SYSTEM.NUM_WORKERS)

    # model
    if cfg.MODEL.MODEL == 'delg_global':
        model = DelgGlobal(num_classes=8050, embedding_size=1024, pretrained=False)  
        model.load_state_dict(torch.load(cfg.EVAL.MODEL_PATH)['model_state_dict'])
    model = model.to(device)
    model = model.eval()
    
    # logging
    ts = time.time()
    logger = setup_logger('testing_info', 'logger/test', file_name='test-log-' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S') + '.txt')
    logger.info(cfg)
    
    gap_m, acc_m = test(cfg, model, df_gallery, df_query, df_sol, gallery_loader, query_loader, args.save_gallery_feats)
    content = 'GAP score: %.5f, Accuracy: %.5f' % (gap_m, acc_m)
    logger.info(content)
    
    
if __name__ == '__main__':
    main()