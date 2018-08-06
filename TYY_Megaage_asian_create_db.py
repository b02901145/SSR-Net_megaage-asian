import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import sys
import dlib
from moviepy.editor import *


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str,
                        help="path to output database mat file")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = './data/megaage_test'
    #output_path = './data/megaage_train'
    img_size = args.img_size

    mypath = './megaage_asian/test'
    #mypath = './megaage_asian/train'
    isPlot = False

    age_file = np.loadtxt('./megaage_asian/list/test_age.txt')
    #age_file = np.loadtxt('./megaage_asian/list/train_age.txt')
    img_name_file = np.genfromtxt('./megaage_asian/list/test_name.txt',dtype='str')
    #img_name_file = np.genfromtxt('./megaage_asian/list/train_name.txt',dtype='str')
    out_ages = []
    out_imgs = []

    for i in tqdm(range(len(img_name_file))):
        
        input_img = cv2.imread(mypath+'/'+img_name_file[i])
        input_img = input_img[20:-20,:,:]
        img_h, img_w, _ = np.shape(input_img)
        age = int(float(age_file[i]))
        if age >= -1:
	        if isPlot:
		        img_clip = ImageClip(input_img)
		        img_clip.show()
		        key = cv2.waitKey(1000)

	        input_img = cv2.resize(input_img,(img_size,img_size))
	        #only add to the list when faces is detected
	        out_imgs.append(input_img)
	        out_ages.append(int(age))

    np.savez(output_path,image=np.array(out_imgs), age=np.array(out_ages), img_size=img_size)

if __name__ == '__main__':
    main()
