# Step-1 | Import all the Dependencies and Libraries Required
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from tqdm import tqdm

# Step - 2 Define A function to provide with a gaussian_fiter

def gausian_filter_density(gt):
    print (gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density

# Step - 3 Standerdise Naming Convensions for the File management system

'''
Here, as per requirement there are 2 pre-requisites "IMG_" convention in the front of the image
and as always a JPG image 

with the middle area it is preferred to consider it with class name/ID and the time that picture took place

i.e. :- IMG_CE217_DDMMYY_HHMMSS.JPG

folder structure for the images would be classified as the use of the images

1. Prepare Folder
    - ALl the dataset captured by the camera in raw image form
2. Examine Folder
    - The images Passed on by the [[Prepare Folder]] to convert as .mat and .h5 and .json formated file
    
    the content of this folder will have naming conventions with a little bit of differ
    there will be two copies of each images from the [[Prepare Folder]]
    1. Astounding with IMG_ format in the fists of the name
    2. Astounding with GT_IMG_ containing ground trouth value of images with mat extension
'''

