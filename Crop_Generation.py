"""
- Crop_generation script is used to crop m x m dimensional sub-images from an original image
- For each generated crop white balance opearation is applied (in order to adjusts the color balance)
- Using ground truth images (binary masks) we can label the crops

Created on April 14 13:40:28 2020

@author: Balazs Horvath
"""

import os
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import io, color, util
import numpy as np
from PIL import Image
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")

def white_balance_LAB(img):
    """
    White balance algoritm in LAB color space. The function has a role in ba
    """
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result

def get_file_names(path):
    """    
    Function that return the filenames from a directory (path is given as input parameter for the function)  
    """ 
    filename = []
    for  filenames in os.walk(path):
            filename.append(filenames[2])
            
    for i, file in enumerate(filename[0]):
        filename[0][i] = path + '\\' + filename[0][i]
    
    return filename[0]

def split_image_names(image_names):
    """
    Function with role in verification and splitting the image_names (names of original and mask images)
    """
    rejected = []
    image_original = []
    image_mask = []

    for item in image_names:
        try:
            with Image.open(item) as img:
                if('mask' in item):
                    image_mask.append(item)
                else:
                    image_original.append(item)

        except:
            rejected.append(item)

    print(f'Images_original:  {len(image_mask)}')
    print(f'Images_mask:  {len(image_original)}')
    print(f'Rejects: {len(rejected)}')
    
    return (image_original,image_mask)

def check_pixel(im_mask,i,j):
    """
    Function checks if pixel on coordinates i,j is part of the mask or not (used for splitting the crops based on classes)
    """
    dimensions = len(im_mask.shape)
    if dimensions == 2:
        return im_mask[i,j]
    else:
        return im_mask[i,j,1]

def crop_generation(im_original,im_mask,nr_image,path_class1,path_class2,name_class1,name_class2,crop_size):
    """
    Function generates crop of dimension crop_size x crop_size x nr_channels (only crop size in defined by user)
    Function also applies White Balance in LAB color space for each image for color balance
    The generated crops are split in two classes (using class_condition function, for our example Hema and Non_hema)
    """
    im_original = np.asarray(im_original)
    im_original = white_balance_LAB(im_original)
    im_mask = np.asarray(im_mask)
    
    class1_nr = 0
    class2_nr = 0
     
    for i in range(crop_size//2,im_mask.shape[0]-crop_size//2,crop_size//2):
        for j in range(crop_size//2,im_mask.shape[1]-crop_size//2,crop_size//2):
            temp_original = im_original[i-crop_size//2:i+crop_size//2,j-crop_size//2:j+crop_size//2,:]
            temp_mask = im_mask[i-crop_size//2:i+crop_size//2,j-crop_size//2:j+crop_size//2]
            
            if check_pixel(im_mask,i,j) :
                crop = Image.fromarray(temp_original)
                crop.save(path_class1+'\\'+name_class1+'_{}_'.format(nr_image)+'{}'.format(class1_nr)+'.jpg')
                class1_nr += 1
            
            else:
                crop = Image.fromarray(temp_original)
                crop.save(path_class2+'\\'+name_class2+'_{}_'.format(nr_image)+'{}'.format(class2_nr)+'.jpg')
                class2_nr += 1


#Example for a directory called Images_of_interests
#Path_class1 and path_class2 are the names of directories where we want to save of crops split in two classes (name_class1,name_class2)
path_images = 'Images_of_interest'
path_class1 = 'Hema_crops'
path_class2 = 'Non_hema_crops'
name_class1 = 'Hema'
name_class2 = 'Non_hema'
crop_size = 28

image_names = get_file_names(path_images)
image_original,image_mask = split_image_names(image_names)

for i in range(0,len(image_original)):
    #print(i)
    im_original = Image.open(image_original[i])
    im_mask = Image.open(image_mask[i])
    crop_generation(im_original,im_mask,i,path_class1,path_class2,name_class1,name_class2,crop_size)
