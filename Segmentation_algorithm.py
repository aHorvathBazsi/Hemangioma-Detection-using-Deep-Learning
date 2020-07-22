import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import cv2 as cv

class CNN_model_crop28(nn.Module):
    """
    A simple Convolutional Neural Network model:
    - 2 convolutional layers, each followed by batch-normalization, ReLU activation and MaxPool
    - 3 fully connected layer, each followed by dropout and ReLU activation
    - Output: logaritmic softmax (you could also use sigmoid for binary classification, but for multi-class classification problems you should use softmax)
    
    Please note that the following model was implemented for an input image having 28x28x3 dimensions
    More general implementation will be updated soon.
    """
    
    def __init__(self): 
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),  # (N, 3, 28, 28) -> (N,  16, 26, 26)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (N, 16, 26, 26) -> (N,  16, 13, 13)
            nn.Conv2d(16, 32, 3, 1), # (N,16,13,13) -> (N, 32, 11, 11)
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # (N,32,11,11) -> (N,32,5,5); 
            # Please note, the final dimension of the features is 32x5x5, that the reason why nn.Linear has 5x5x32 as its first parameter
            # Please note that N is the number of images in a batch (batch_size defined create datasets function)
        )
        self.classifier = nn.Sequential(
            nn.Linear(5*5*32, 120),         # (N, 5x5x32) -> (N, 120)
            nn.Dropout(0.5), 
            nn.ReLU(),
            nn.Linear(120, 84),         # (N, 120) -> (N, 84)
            nn.Dropout(0.5), 
            nn.ReLU(),
            nn.Linear(84,32),         # (N, 84) -> (N, 32)
            nn.Dropout(0.5), 
            nn.ReLU(),
            nn.Linear(32,2)          # (N, 32) -> (N, 2)
       )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 5*5*32)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    
def analiza_pixel(model,crop,transfroms):
    crop = transfroms(crop)
    crop = crop.view(-1,3,28,28)
    y_val = model(crop)
    #print(y_val)
    prediction = torch.max(y_val,1)[1]
    if prediction == 0:
        return 255
    else:
        return 0
    
def white_balance_LAB(img):
    """
    White balance algoritm in LAB color space.
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
    Function that returns the filenames from a directory (path is given as input parameter for the function)  
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

def verify_mask(image_mask):
    
    mask_modified = np.zeros(image_mask.shape).astype(np.uint8)
    
    if(np.max(image_mask) == True):
        mask_modified[image_mask==True] = 255
        mask_modified[image_mask==False] = 0
    
    else:
        mask_modified = np.copy(image_mask)
    
    return mask_modified

def segmentation_algoritm(path,input_image_name,input_mask_name,model,image_transforms):
    """
    Applies pixel-based segmentation to the input image and returns the generated binary-mask and different segmentation metrics.
    
    Input variables: input_image_name, input_mask_name, model, image_transforms
    
    Output: metrics dictionary (contains performance metrics), segmented_image
    """
    
    metrics = {}
    
    input_image = Image.open(input_image_name)
    input_image = np.asarray(input_image)
    input_image_balanced = white_balance_LAB(input_image)
    input_mask = Image.open(input_mask_name)
    input_mask = np.asarray(input_mask)
    
    segmented_image = np.zeros((input_image.shape[0],input_image.shape[1])).astype(np.uint8)
    
    #plt.figure(figsize=(6,6))
    #plt.imshow(input_image)
    #plt.figure(figsize=(6,6))
    #plt.imshow(input_image_balanced)
    #plt.figure(figsize=(6,6))
    #plt.imshow(input_mask,cmap='gray')
    
    n = 1
    for i in range (14,input_image_balanced.shape[0]-14,2):
        for j in range(14,input_image_balanced.shape[1]-14,2):
            crop = input_image_balanced[i-14:i+14,j-14:j+14,:]
            crop = Image.fromarray(crop)
            if(n%5000 == 0):
                print(n)
            n+=1
            segmented_image[i:i+2,j:j+2] = analiza_pixel(model,crop,image_transforms)
            
    #plt.figure(figsize=(6,6))
    #plt.imshow(segmented_image,cmap='gray')
    image_output = Image.fromarray(segmented_image)
    output_path = input_mask_name.split('.')[0]+'_segmented.tiff'
    image_output.save(output_path)
    
    input_mask = verify_mask(input_mask)
    
    if(len(input_mask.shape)==3):
        mask_verification = input_mask[:,:,1]
    else:
        mask_verification = input_mask
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(0,segmented_image.shape[0]):
        for j in range(0,segmented_image.shape[1]):
            if mask_verification[i,j] == 255 and segmented_image[i,j]==255:
                TP += 1
            elif mask_verification[i,j] == 255 and segmented_image[i,j]==0:
                FN += 1
            elif mask_verification[i,j] == 0 and segmented_image[i,j]==255:
                FP += 1
            else:
                TN += 1
    
    Sensitivity = TP/(TP+FN)*100
    Specificity = TN/(TN+FP)*100
    PPV = TP/(TP + FP)*100
    NPV = TN/(TN + FN)*100
    
    metrics["Sensitivity"] = Sensitivity
    metrics["Specificity"] = Specificity
    metrics["PPV"] = PPV
    metrics["NPV"] = NPV
    
    print(input_image_name + ' image is done!')
    
    return metrics

model = CNN_model_crop28()
model.load_state_dict(torch.load('Hemagioma_dataset82_epoch_60.pt'))
model.eval()

image_transforms = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

path = 'database_82_segmented_model_vechi'
filenames = get_file_names(path)

image_names,mask_names = split_image_names(filenames)
#image_names
#mask_names

Specificity = []
Sensitivity = []
PPV = []
NPV = []

for i in range(len(image_names)):
    metrics = segmentation_algoritm(path,image_names[i],mask_names[i],model,image_transforms)
    Specificity.append(metrics["Specificity"])
    Sensitivity.append(metrics["Sensitivity"])
    PPV.append(metrics["PPV"])
    NPV.append(metrics["NPV"])
    
segmentation_results = pd.DataFrame(list(zip(Specificity, Sensitivity,PPV,NPV)), columns =['Specificity', 'Sensitivity','PPV','NPV'],index=image_names)

segmentation_results.to_csv('Segmentation_results_test_database82.csv')
