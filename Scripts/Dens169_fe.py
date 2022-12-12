from keras.preprocessing import image
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import pyplot as plt
import cv2
import gc
import tensorflow as tf
from tqdm import tqdm
from keras.models import Sequential
from keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet169
from keras.layers import merge, Input


image_input = Input(shape=(224,224,3))

model = DenseNet169(include_top=False,weights="imagenet",input_tensor=image_input)

model.summary()

data_dir = os.listdir("/content/drive/MyDrive/original_Images/Training_Set")

densenet169_feature_list=[]
for i in data_dir:
  img_path ="/content/drive/MyDrive/original_Images/Training_Set" +"/"+i
  img = image.load_img(img_path, target_size=(224, 224))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)

  densenet169_feature = model.predict(img_data)
  densenet169_feature_np = np.array(densenet169_feature)
  densenet169_feature_list.append(densenet169_feature_np.flatten())

densenet169_feature_list_np = np.array(densenet169_feature_list)

np.save("densenet169_features",densenet169_feature_list_np)

import glob

images_dir = '/content/drive/MyDrive/original_Images/Training_Set/'
image_list = glob.glob(images_dir + '*')

# Create array of filenames without .jpeg to join with ground-truth classification
keys1 = [os.path.splitext(os.path.basename(image))[0] for image in image_list]

# Save to disk
np.save("densenet169_features",densenet169_feature_list_np)
np.save('keys1', keys1)

densenet169_feature_list_np.shape

keys1 = np.load('keys1.npy')
features1 = np.load('densenet169_features.npy')
labels_df = pd.read_csv('/content/drive/MyDrive/Groundtruth_labels/Disease_Grading_Training_Labels.csv', index_col='image_name')

# Check lengths are the same for all raw data
len(keys1), len(features1), len(labels_df)

features_df = pd.DataFrame(features1, index=keys1)
df = pd.merge(features_df, labels_df, how='inner', left_index=True, right_index=True)

df.head(4)

df.level.hist(figsize=(4,3));
df.level.value_counts()

min_class = df.level.value_counts().min() min_class

feature_cols = df.columns[:-10]  
X = df[feature_cols].values
y = df['level'].values

X.shape, y.shape



