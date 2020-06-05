"""
Shuffle_directory is used to shuffle your data: after you generate your crops, it is highly recommended to shuffle your data
before you create a dataset (train_test_split could solve the problem, but in this proiect we use Pytorch framework, thus
we would not use any of the sklearn functions; more details are presented in the Hemangioma_classification.py script)
"""

import os
import matplotlib.pyplot as plt
from skimage import io, color, util
import numpy as np
from PIL import Image
from IPython.display import display

# Filter harmless warnings
import warnings
warnings.filterwarnings("ignore")

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

def shuffle_directory(path_original,path_shuffled):
    """
    The function actually copies all files fom path_original to path_shuffled, but using different order of files.
    Using the number of files within the directory, a vector (having the same length as number of file) is generated
    and shuffled. In this way a new order is generated and based on this order the files are copied to the new directory.
    """
    img_names = get_file_names(path_original) # getting the filenames from the original path
    
    order = np.arange(0,len(img_names)) # creating a vector (each element represents the number of a specific filename)
    np.random.shuffle(order) # shuffle the order of files (np.random.shuffle is an inplace function)
    
    # write the files to the new directory using the shuffled order
    # rejected checks whether there is a file that can not be opened using Image.open()
    rejected = 0
    for i in range(len(order)):
        try:
            with Image.open(img_names[order[i]]) as img:
                img = np.asarray(img)
                img = Image.fromarray(img)
                img.save(path_shuffled+'\\'+'{}'.format(i+1)+'.jpg')

        except:
            rejected += 1
    print(f'Rejects: {rejected}')
                                                      
# Example: shuffle two directories Hema_crops and Non_hema_crops (The shuffled data is stored in Shuffled_hema_crops and Shuffled_non_hema_crops)
                                                      
path_original_class1 = 'Hema_crops'
path_shuffle_class1 = 'Shuffled_hema_crops'
path_original_class2 = 'Non_hema_crops'
path_shuffle_class2 = 'Shuffled_non_hema_crops'
                                                      
#shuffle_directory(path_original_class1,path_shuffle_class1)
shuffle_directory(path_original_class2,path_shuffle_class2)
