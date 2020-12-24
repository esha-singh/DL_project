# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 11:23:56 2020

@author: tedkuo
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class AttentionModule(nn.Module):
    def __init__(self, num_classes):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.softplus = nn.Softplus()
        self.attn_classifier = nn.Linear(1024, num_classes)
        
        nn.init.normal_(self.attn_classifier.weight)
        
    def forward(self, reconstructed_f):
        x = self.conv1(reconstructed_f)
        x = self.bn1(x)
        x = self.relu(x)
        
        score = self.conv2(x)
        score = self.bn2(score)
        prob = self.relu(score)
        
        norm_recon_f = F.normalize(reconstructed_f, dim=1)
        #feats = norm_local_f*prob
        feats = torch.mean(torch.mul(norm_recon_f, prob), [2, 3], keepdims=False)
        #print(feats.shape)
        attn_logits = self.attn_classifier(feats)
        
        return attn_logits, feats, score, prob
   
    
class Delg(nn.Module):
    def __init__(self, num_classes, embedding_size=2048, pretrained=True):
        super(Delg, self).__init__()
        resnet = models.resnet.resnet50(pretrained=pretrained)
        self.backbone_to_conv4 = nn.Sequential(*list(resnet.children()))[:-3]
        self.backbone_conv5 = nn.Sequential(*list(resnet.children()))[-3]
        self.gem_pool = GeM()
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_size) # backbone_out_feature_size not sure. probably 2048
        self.prelu = nn.PReLU()
        self.arcface = ArcFace(embedding_size, num_classes)
        self.attention = AttentionModule(num_classes)
        #self.autoencoder = AutoEncoder()
        
        nn.init.normal_(self.embedding.weight)
        
    def forward(self, image, labels=None, global_only=False, training=True):
        local_f = self.backbone_to_conv4(image)
        global_f = self.backbone_conv5(local_f)
        global_f = self.gem_pool(global_f)
        global_f = global_f.view(global_f.size(0), -1)
        global_f = self.embedding(global_f)
        global_f = self.prelu(global_f)
        global_logits = self.arcface(global_f, labels, training)
        
        if not global_only:
            #reconstructed_f, encoded_f = self.autoencoder(local_f)
            attn_logits, attn_f, score, prob = self.attention(local_f)   
            return F.normalize(global_f), global_logits, F.normalize(attn_f), attn_logits, score, prob
        
        else:
            return F.normalize(global_f), global_logits
        
    
    
