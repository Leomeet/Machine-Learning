import datetime
import cv2
import subprocess
import os
import moviepy.editor as moviepy

# file format of training videos
'''
Training Data > image-Time.img
'''
cwd = os.getcwd()
parent_path_dir = os.path.join(cwd, "Training Data") # Parent in this case is folder "Training Data"

def get_path():
    e = datetime.datetime.now()
    global parent_path_dir
    image_base = "image-"+ str(e.day) + str(e.month) + str(e.year) + str(e.hour) + str(e.minute) + str(e.second)
    path_dir = os.path.join(parent_path_dir,image_base)
    return path_dir

def extract(video_path):
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(get_path()+'.png',frame)
    cap.release()
    cv2.destroyAllWindows()


def get_file_paths(base_dir):

    dir_list = os.listdir(base_dir)
    for j in range(len(dir_list)):

        path = os.path.join(base_dir,dir_list[j])
        dir_list[j] = path


    return dir_list


# "W://_00_ Projects//Machine Learning training videos" path for training videos
if __name__ == "__main__":

    videos = get_videos("W:\_00_ Projects\Machine Learning training videos")
    # for i in range(1,len(videos)):
    #     print(videos[i])
    #     extract(videos[i])