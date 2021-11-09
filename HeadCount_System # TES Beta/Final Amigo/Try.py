import h5py
import matplotlib
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from tqdm import tqdm
import settings
from image import *
from model import CSRNet
import torch

# %matplotlib inline

from torchvision import datasets, transforms
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])

# Creating a General List of the images that we have collected
root = os.path.dirname(os.path.abspath(__file__))

# All the raw images resides in a different folder here named as :prepared
prepared = os.path.join(root,"images" ,"prepared")
path_sets = [prepared]
img_paths = []

counter = 0
for dir in path_sets:
    img_paths.append(settings.get_files(dir,"jpg"))


print(img_paths[0])

