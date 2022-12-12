# Importing Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# Function to Resize the Images
def resize(img):
    ratio  = min([1152/img.shape[0], 1500/img.shape[1]])
    return cv2.resize(img,(int(img.shape[1]*ratio),int(img.shape[0]*ratio)), interpolation = cv2.INTER_CUBIC)

# Function for Grayscale Transformation
def rgb2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Function for Visualising the Images
def imshow(img):
    plt.axis('off')
    plt.imshow(img, 'gray')
    plt.show()

# Function for Adaptive Histogram Equalization
def clahe_equalized(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    return  cl1

# Evaluation Metrics - To Measure the Sensitivity
def evaluation(image, mask):
    
    zeros_list_img, one_list_img, zeros_list_mk, one_list_mk = [], [], [], []
    
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            val_mk = mask[i][j]
            val_img  = image[i][j]
            if val_mk == 0:
                zeros_list_mk.append((i,j))
            else:
                one_list_mk.append((i,j))
            if val_img == 0:
                zeros_list_img.append((i,j))
            else:
                one_list_img.append((i,j))

    TP = len(set(one_list_img).intersection(set(one_list_mk)))
    TN = len(set(zeros_list_img).intersection(set(zeros_list_mk)))
    FP = len(set(one_list_img).intersection(set(zeros_list_mk)))
    FN = len(set(zeros_list_img).intersection(set(one_list_mk)))
    R = TP/(TP + FN)
    return R

# Image and Label Path
image_path = "/content/drive/MyDrive/Segmentation/OriginalImages/Training Set/"
hm_labels_path = "/content/drive/MyDrive/Segmentation/All Segmentation Groundtruths/Training Set/Haemorrhages/"
images = os.listdir(image_path)
hm_labels = os.listdir(hm_labels_path)
images.sort()

for img_number in range(55,82,1):

  if ((img_number>=1) and (img_number<=9)):
        s = '0' + str(img_number)
  else:
        s = str(img_number)

# Reading the Image
img_file = image_path + "Filename" + s + ".jpg"
img = cv2.imread(img_file)

# Reading Mask
hm_file = hm_labels_path + "Filename" + s + "_HE.tif"
img_hm = cv2.imread(hm_file)

# Label
import cv2
img_hm = cv2.cvtColor(img_hm,cv2.COLOR_RGB2GRAY)
T, img_hm = cv2.threshold(img_hm, 0, 255, cv2.THRESH_BINARY)

# Fundus Images
img = resize(img)
gray  = rgb2gray(img)
b,g,r = cv2.split(img)
img_enhanced = clahe_equalized(g)

# Pipeline 1 - ksize 81
img_medf = cv2.medianBlur(img_enhanced,81)
img_sub = cv2.subtract(img_medf,img_enhanced)
img_subf = cv2.blur(img_sub,(5,5))
ret, img_darkf = cv2.threshold(img_subf, 10, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
img_darkl = cv2.morphologyEx(img_darkf,cv2.MORPH_OPEN,kernel)

# Pipeline 2 - ksize 131
img_medf1 = cv2.medianBlur(img_enhanced,131)
img_sub1 = cv2.subtract(img_medf1,img_enhanced)
img_subf1 = cv2.blur(img_sub1,(5,5))
ret, img_darkf1 = cv2.threshold(img_subf1, 10, 255, cv2.THRESH_BINARY)
img_darkl1 = cv2.morphologyEx(img_darkf1,cv2.MORPH_OPEN,kernel)

# Bitwise Operations
img_both = cv2.bitwise_or(img_darkl,img_darkl1)

# RResizing to Original
result = cv2.resize(img_both, (4288,2848), interpolation=cv2.INTER_CUBIC)

# # Matching and Evaluating the Results
print('Haemorrhages Detection: ')
imshow(result)

# print('Haemorrhages Labels: ')
imshow(img_hm)

# print('------------------------')
R = evaluation(result, img_hm)
print('Sensitivity = ', end = '')
print(R*100)
