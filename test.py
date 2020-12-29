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
from matching import get_inliers
from extract_local_features import delf_extract
from model import Delg


def test(cfg, model, df_gallery, df_query, df_sol, gallery_loader, query_loader, save_gallery_feats=False):
    
    device = cfg.SYSTEM.DEVICE
    batch_size = cfg.EVAL.BATCH_SIZE
    
    GLOBAL_TOP_K = 5
    LOCAL_TOP_K = 100
    #CLS_TOP_K = 5
    #attn_threshold = 0.01
    with torch.no_grad():
        """
        Compute global descriptor for gallery set
        """
        global_feats = []
        gallery_kp_locations_list = []
        gallery_descriptors_list = []
        #gallery_scores_list = []
        if save_gallery_feats:
            for img_0, img_1, img_2 in tqdm(gallery_loader): 
                img_0, img_1, img_2 = img_0.to(device), img_1.to(device), img_2.to(device)
                
                #feat_0, _ = model(img_0, training=False)
                global_f, _, _, _, encoded_f, _, _, prob = model(img_1, labels=None, global_only=False, training=False)
                global_f = F.normalize(global_f)
                """
                gallery_kp_locations = []
                gallery_scores = []
                gallery_descriptors = []
                #print(prob.shape)
                #for k in range(batch_size):
                for i in range(prob.shape[2]):
                    for j in range(prob.shape[3]):
                        #print(prob[:, :, i, j])
                        if prob[:, :, i, j].detach().cpu() > attn_threshold:
                            gallery_kp_locations.append((i, j))
                            gallery_scores.append(prob[:, :, i, j].detach().cpu()) 
            
                for loc in gallery_kp_locations:
                    #encoded_f = F.normalize(encoded_f)
                    gallery_descriptors.append(encoded_f[:, :, loc[0], loc[1]].detach().cpu().numpy())
                """
                gallery_kp_locations, gallery_descriptors, gallery_scores = delf_extract(img_1, model)
                #feat_2, _ = model(img_2, training=False)
                
                #feat = (feat_0 + feat_1 + feat_2)/3
                #feat = torch.cat([feat_0, feat_1, feat_2], dim=1)
                #feat = F.normalize(feat)
                #feat = torch.cat([feat_0, feat_1, feat_2], dim=1)
                
                global_feats.append(global_f.detach().cpu())
                gallery_kp_locations_list.append(gallery_kp_locations)
                gallery_descriptors_list.append(gallery_descriptors)
                #gallery_scores_list.append(gallery_scores)
            """   
            with open('gallery_kp_locations_list.txt', 'w') as f:
                for item in gallery_kp_locations_list:
                    f.write("%s\n" % item)
            with open('gallery_descriptors_list.txt', 'w') as f:
                for item in gallery_descriptors_list:
                    f.write("%s\n" % item)
            """
            global_feats = torch.cat(global_feats)
            torch.save(global_feats, 'gallery_feats_' + str(cfg.DATASET.IMAGE_SIZE) + '.pt')
              
        GLOBAL_ONLY_PROBS = []
        GLOBAL_ONLY_PREDS = []
        PROBS = []
        PREDS = []
        
        #PRODS_M = []
        #PREDS_M = []      
        global_feats = torch.load('gallery_feats_' + str(cfg.DATASET.IMAGE_SIZE) + '.pt')
        print(global_feats.shape)
        global_feats = global_feats.to(device)
        
        """
        Compute global descriptor for query set
        """
        for img_0, img_1, img_2 in tqdm(query_loader):
            img_0, img_1, img_2 = img_0.to(device), img_1.to(device), img_2.to(device)
            
            #feat_0, logits_0 = model(img_0, training=False)
            global_f, _, _, _, encoded_f, _, _, prob = model(img_1, labels=None, global_only=False, training=False)
            global_f = F.normalize(global_f)
            """
            query_kp_locations = []
            query_scores = []
            query_descriptors = []
            #print(prob.shape)
            #for k in range(batch_size):
            for i in range(prob.shape[2]):
                for j in range(prob.shape[3]):
                    #print(prob[:, :, i, j])
                    if prob[:, :, i, j].detach().cpu() > attn_threshold:
                        query_kp_locations.append((i, j))
                        query_scores.append(prob[:, :, i, j].detach().cpu())
            
            for loc in query_kp_locations:
                #encoded_f = F.normalize(encoded_f)
                query_descriptors.append(encoded_f[:, :, loc[0], loc[1]].detach().cpu().numpy())
            """
            #feat_2, logits_2 = model(img_2, training=False)
            #feat = (feat_0 + feat_1 + feat_2)/3
            #feat = torch.cat([feat_0, feat_1, feat_2], dim=1)
            #feat = F.normalize(feat)
            
            # cosine similarity of the current query image to the images in the gallery set
            similarity = global_f.mm(global_feats.t())
            #sim_mean = torch.mean(similarity, dim=1, keepdim=True)
            #similarity *= sim_mean
            #similarity = F.softmax(similarity, dim=1)
            
            # Retrieve top-k images from the train set
            (global_only_values, global_only_indices) = torch.topk(similarity, GLOBAL_TOP_K, dim=1)
            (values, indices) = torch.topk(similarity, LOCAL_TOP_K, dim=1)
            
            query_kp_locations, query_descriptors, query_scores = delf_extract(img_1, model)
            
            #print(indices.shape)
            prediction_score_list = []
            for value, idx in zip(values[0], indices[0]):
                #(idx)
                num_inliers, _, _ = get_inliers(query_kp_locations, query_descriptors, gallery_kp_locations_list[idx], gallery_descriptors_list[idx])
                #print(num_inliers)
                score = min(num_inliers, 50)/50 + 0.25*value
                prediction_score_list.append(score)
                
            #values = F.softmax(values, dim=1)
            global_only_probs = global_only_values
            global_only_preds = global_only_indices # indices in the gallery set
            GLOBAL_ONLY_PROBS.append(global_only_probs.detach().cpu())
            GLOBAL_ONLY_PREDS.append(global_only_preds.detach().cpu())
            
            PROBS.append(max(prediction_score_list))
            PREDS.append(indices[0][prediction_score_list.index(max(prediction_score_list))].detach().cpu().numpy())

        GLOBAL_ONLY_PROBS = torch.cat(GLOBAL_ONLY_PROBS).numpy()
        GLOBAL_ONLY_PREDS = torch.cat(GLOBAL_ONLY_PREDS).numpy()
        PROBS = np.array(PROBS)
        PREDS = np.array(PREDS)
        #PREDS = np.array(PREDS)
        
        # Convert the gallery set indices to landmark ids
        gallery_landmark_id = df_gallery['landmark_id'].values
        GLOBAL_ONLY_PREDS = gallery_landmark_id[GLOBAL_ONLY_PREDS]
        PREDS = gallery_landmark_id[PREDS]
        
        
        """
        Get the final prediction of the landmark with the highest score from the top-k retrieved images
        """
        GLOBAL_ONLY_PROBS_F = []
        GLOBAL_ONLY_PREDS_F = []
        for i in tqdm(range(GLOBAL_ONLY_PREDS.shape[0])):
            tmp = {}
            for k in range(GLOBAL_TOP_K):
                pred_k = GLOBAL_ONLY_PREDS[i, k]
                tmp[pred_k] = tmp.get(pred_k, 0.) + float(GLOBAL_ONLY_PROBS[i, k])
            
            pred, conf = max(tmp.items(), key=lambda x: x[1])
            GLOBAL_ONLY_PREDS_F.append(pred)
            GLOBAL_ONLY_PROBS_F.append(conf)
        
        PROBS_F = PROBS
        PREDS_F = PREDS
        #print(PROBS_F)
        #print(GLOBAL_ONLY_PREDS_F)
        
        y_true = {idx: target for idx, target in enumerate(df_query['landmark_id'])}
        global_only_pred_f = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(GLOBAL_ONLY_PREDS_F, GLOBAL_ONLY_PROBS_F))}
        global_only_gap_m, global_only_acc_m = global_average_precision_score_test(y_true, global_only_pred_f)
        
        pred_f = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_F, PROBS_F))}
        gap_m, acc_m = global_average_precision_score_test(y_true, pred_f)
        
        return global_only_gap_m, global_only_acc_m, gap_m, acc_m
        
    
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

    
    #df_query = df_query.head(5000)
    df_gallery = df_gallery.head(100000)
    
    #Eliminate the landmarks that are not in the gallery (train) set
    arr = df_gallery['landmark_id'].values
    arr = list(dict.fromkeys(arr))
    drop = []
    for idx, row in df_query.iterrows():
        bool_list = []
        if row['landmark_id'] is not None:
            for landmark in row['landmark_id']:
                bool_list.append(landmark in arr)
            if not any(bool_list):
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
    if cfg.MODEL.MODEL == 'delg':
        model = Delg(num_classes=8050, embedding_size=1024, pretrained=False)  
        model.load_state_dict(torch.load(cfg.EVAL.MODEL_PATH)['model_state_dict'])
    model = model.to(device)
    model = model.eval()
    
    # logging
    ts = time.time()
    logger = setup_logger('testing_info', 'logger/test', file_name='test-log-' + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S') + '.txt')
    logger.info(cfg)
    
    global_only_gap_m, global_only_acc_m, gap_m, acc_m = test(cfg, model, df_gallery, df_query, df_sol, gallery_loader, query_loader, args.save_gallery_feats)
    content = 'Global GAP Score: %.5f, Global Accuracy: %.5f, GAP Score: %.5f, Accuracy: %.5f' % (global_only_gap_m, global_only_acc_m, gap_m, acc_m)
    logger.info(content)
    
    
if __name__ == '__main__':
    main()