# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 11:23:56 2020

@author: tedkuo
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from resnet import resnet50
import torchvision.models as models


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        #self.p = nn.Parameter(torch.ones(1)*p)
        self.p = p
        self.eps = eps
    
    def forward(self, x):
        return self.gem(x, p=self.p, eps = self.eps)
    
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    
    
class ArcFace(nn.Module):
    def __init__(self, embedding_size, num_classes, scale_factor=45.25, arcface_margin=0.1):
        super(ArcFace, self).__init__()
        self.num_classes = num_classes
        self.scale_factor = nn.Parameter(torch.ones(1)*scale_factor)
        self.arcface_margin = arcface_margin
        #self.cosine_weights = cosine_weights
        self.cosine_weights = nn.Parameter(torch.FloatTensor(embedding_size, num_classes))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.cosine_weights.size(1))
        self.cosine_weights.data.uniform_(-stdv, stdv)
        
    def forward(self, global_features, labels=None, training=True):
        """
        Parameters
        ----------
        global_features : float tensor with shape [batch_size, embedding_size]
        
        Returns
        -------
        logits : float tensorwith shape [batch_size, num_classes]
        """
        norm_global_features = F.normalize(global_features, dim=1)
        norm_cosine_weights = F.normalize(self.cosine_weights, dim=0)
        cosine_sim = torch.mm(norm_global_features, norm_cosine_weights)
        
        if training and self.arcface_margin > 0:
            one_hot_labels = F.one_hot(labels, self.num_classes)
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
        selected_labels = torch.where(torch.gt(theta, math.pi - self.arcface_margin),
                                      torch.zeros_like(one_hot_labels),
                                      one_hot_labels)
        final_theta = torch.where(selected_labels.type(torch.bool),
                                  theta + self.arcface_margin,
                                  theta)
        cosine_sim_with_margin = torch.cos(final_theta)
        return cosine_sim_with_margin
        
    
class DelgGlobal(nn.Module):
    def __init__(self, num_classes, embedding_size=2048, pretrained=True):
        super(DelgGlobal, self).__init__()
        resnet = models.resnet.resnet50(pretrained=pretrained)
        self.backbone_to_conv4 = nn.Sequential(*list(resnet.children()))[:-3]
        self.backbone_conv5 = nn.Sequential(*list(resnet.children()))[-3]
        #self.backbone = resnet50(pretrained=pretrained)
        self.gem_pool = GeM()
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_size) # backbone_out_feature_size not sure. probably 2048
        #self.cosine_weights = nn.Parameter(torch.rand(embedding_size, num_classes))
        self.arcface = ArcFace(embedding_size, num_classes)
        
        nn.init.normal_(self.embedding.weight)
        
    def forward(self, image, labels=None, training=True):
        local_f = self.backbone_to_conv4(image)
        global_f = self.backbone_conv5(local_f)
        x = self.gem_pool(global_f)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        logits_m = self.arcface(x, labels, training)
        
        return F.normalize(x), logits_m
        
        
    
    
