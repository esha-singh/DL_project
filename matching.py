import os, sys, time

import numpy as np
from PIL import Image
import io
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform

import cv2

_DISTANCE_THRESHOLD = 0.8          # Adjust this value depending on your dataset. 
                                    # This value needs to be engineered for optimized result.

def get_inliers(locations_1, descriptors_1, locations_2, descriptors_2):  
    locations_1 = np.array(locations_1)
    descriptors_1 = np.array(descriptors_1)
    descriptors_1 = np.squeeze(descriptors_1)
    locations_2 = np.array(locations_2)
    descriptors_2 = np.array(descriptors_2)
    descriptors_2 = np.squeeze(descriptors_2)
    
    #print(locations_1.shape)
    #print(descriptors_1.shape)
    #print(locations_2.shape)
    #print(descriptors_2.shape)
    
    num_features_1 = locations_1.shape[0]
    num_features_2 = locations_2.shape[0]

    # Find nearest-neighbor matches using a KD tree.
    if len(descriptors_1.shape) != 2 or len(descriptors_2.shape) != 2:
        num_inliers = 0
        return num_inliers, None, None
    
    d1_tree = cKDTree(descriptors_1)
    distances, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)
    
    if isinstance(indices, int):
        num_inliers = 0
        return num_inliers, None, None

    # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        locations_2[i,] for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],] for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    
    #print(locations_1_to_use)
    #print(locations_2_to_use)
    
    if len(locations_1_to_use) < 1 or len(locations_2_to_use) < 1:
        num_inliers = 0
        return num_inliers, locations_1_to_use, locations_2_to_use
    
    # Perform geometric verification using RANSAC.
    model_robust, inliers = ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)
    
    num_inliers = sum(inliers)
            
    return num_inliers, locations_1_to_use, locations_2_to_use
