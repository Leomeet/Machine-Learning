'''
Renaming file with theire finite file convension

this function will make sure that files from Dir (A) to Focused Dir

Focused Dir:
Focused dir contains the files to prepare for later
sort to speak row files to prepare later

Use:
this function will be used while taking the account of files that comes from External sources (i.e.: Camara / Server)

Input:
File dir, Time, Collage ID/ Name , class id

NOTE: in case of the files with predefined naming convention
with our recommended properties the file will just change the dir path
    - for that there will only be 3 items needed
        1. Root Dir
        2. Collage ID/Name
        3. Class Id

'''

# -----------
import glob, os, warnings
import shutil
import sys
import glob
import h5py
import cv2
import numpy as np
import scipy
import scipy.io
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.image as image
import matplotlib.pyplot as plt


class OUF(Exception):
    pass


class Settings:

    def __init__(self):
        global ROOT
        dir = os.path.abspath(os.getcwd())
        ROOT = dir
        # print("Current Root Dir =>"+dir)

    def select_files(self):
        pass


def get_files(root_dir, type):
    type = r"*." + type
    file_list = glob.glob(os.path.join(root_dir, type))

    if len(file_list) == 0:
        raise OUF("[INFO:Error] No files are detected")
    else:
        print(f"\n[Files]: {len(file_list)} Files")

    return file_list


def gaussian_filter_density(gt):
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


def copy_paste(file_list):
    path = "W:\_00_ Projects\# Pycharm\TES Beta # Machine Learning\Final_Amigo\Images\Prepared"
    destination = Path(path)
    a = len(file_list)
    for i in file_list:
        shutil.copy(i, destination)

    b = get_files(destination, "jpg")
    b = len(b)
    if a == b:
        print("[INFO] Transaction of files: successful..!!")
    else:
        warnings.warn(f"{abs(a - b)} Files are Failed to copy")


def convert_jpg(root_dir):
    file_list = get_files(root_dir, "png")

    for i in file_list:
        img = Image.open(i)
        rgb_image = img.convert('RGB')
        temp_i = i.replace('.png', '.jpg')
        rgb_image.save(temp_i)

    file_list_jpg = get_files("jpg")

    if len(file_list_jpg) != len(file_list):
        diff = abs(len(file_list_jpg) - len(file_list))
        warnings.warn(f"[Warning]: Some files are not converted")
        print(f"####### {diff} files are not processed")

    else:
        print(f"[INFO]: Conversion Successful")


def create_HDF5(img,dest_path):

    filename = os.path.basename(img)
    with h5py.File(os.path.join(dest_path, filename+'.h5'),'w') as hdf:
        hdf.create_dataset(img , data=img)

def creat_mat(img,dest_path):

    filename = os.path.splitext(os.path.basename(img))[0]
    print(filename)
    img_array = plt.imread(img,3)
    scipy.io.savemat(os.path.join(dest_path, filename+".mat"), {"image":img_array})



if __name__ == "__main__":
    settings = Settings()
    Base_dir = r"W:\_00_ Projects\# Pycharm\TES Beta # Machine Learning\Final_Amigo\Images\prepared\image-19920212270.jpg"
    Dest_dir = r"W:\_00_ Projects\# Pycharm\TES Beta # Machine Learning\Final_Amigo\Images\examin"

    creat_mat(Base_dir,Dest_dir)

