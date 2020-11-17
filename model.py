# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 11:23:56 2020

@author: tedkuo
"""
import math
import torch
import torch.nn as nn
import torch.functional as F
from resnet import resnet50


class GeM(nn.module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    
    def forward(self, x):
        return self.gem(x, p=self.p, eps = self.eps)
    
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x,size(-1))).pow(1./p)
    
    
class ArcFace(nn.Module):
    def __init__(self, cosine_weights, labels, num_classes, training=True, scale_factor=45, arcface_margin=0.1):
        self.training = training
        self.labels = labels  
        self.num_classes = num_classes
        self.scale_factor = scale_factor
        self.arcface_margin = arcface_margin
        self.cosine_weights = cosine_weights
        
    def forward(self, global_features):
        """
        Parameters
        ----------
        global_features : float tensor with shape [batch_size, embedding_size]
        
        Returns
        -------
        logits : float tensorwith shape [batch_size, num_classes]
        """
        norm_global_features = F.normalize(global_features)
        norm_cosine_weights = F.normalize(self.cosine_weights)
        cosine_sim = torch.mm(norm_cosine_weights, norm_global_features)
        
        if self.training and self.arcface_margin > 0:
            one_hot_labels = F.one_hot(self.labels, self.num_classes)
            cosine_sim = self.apply_arcface_margin(cosine_sim, one_hot_labels)
        
        logits = self.scale_factor*cosine_sim
        
        return logits
    
    def apply_arcface_margin(self, cosine_sim, one_hot_labels):
        """
        Parameters
        ----------
        cosine_sim: float tensor with shape [batch_size, num_classes]
        one_hot_labels: int tensor with shape [batch_size, num_classes]

        Returns
        -------
        cosine_sim_with_margin: float tensor with shape [batch_size, num_classes]
        """
        theta = torch.acos(cosine_sim)
        selected_labels = torch.where(torch.greater(theta, math.pi - self.arcface_margin),
                                      torch.zero_like(one_hot_labels),
                                      one_hot_labels)
        final_theta = torch.where(selected_labels.type(torch.bool),
                                  theta + self.arcface_margin,
                                  theta)
        cosine_sim_with_margin = torch.cos(final_theta)
        return cosine_sim_with_margin
        
    
class DelgGlobal(nn.module):
    def __init__(self, labels, num_classes, embedding_size, pretrained=True):
        super(DelgGlobal, self).__init__()
        self.labels = labels
        self.num_classes = num_classes
        self.backbone = resnet50(pretrained=pretrained)
        self.gem_pool = GeM()
        self.arcface = ArcFace(self.cosine_weights, self.labels, self.num_classes)
        self.embedding = nn.Linear(backbone_out_feature_size, embedding_size)
        self.cosine_weights = nn.Linear(embedding_size, num_classes)
    
    def forward(self, image):
        _, global_f = self.backbone(image)
        x = self.gem_pool(global_f)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.arcface(x)
        
        return x
        
        
    
    
