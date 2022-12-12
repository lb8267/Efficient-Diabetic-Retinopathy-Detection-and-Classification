import tensorflow as tf
from tensorflow import lite

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import ResNet50,inception_v3,nasnet
from sklearn.utils import class_weight
import os
import glob

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np

import shutil

from tensorflow.compat.v1 import ConfigProto,InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model_dir = '/content/drive/MyDrive/NASNet-mobile.h5'

images_dir = '/content/drive/MyDrive/original_Images/Training_Set/'
image_list = glob.glob(images_dir + '*')

print('Found %d images' % len(image_list))

def create_graph():
    
    
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def extract_features(image_list):
    '''Extract deep features from images in image_list.'''
    
    nb_features = 2048
    features = np.empty((len(image_list), nb_features))

    create_graph()

    with tf.compat.v1.Session() as sess:
        bottleneck_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        print('Processing %i images' % len(image_list))
        for idx, image in enumerate(image_list):
            
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(bottleneck_tensor, {'DecodeJpeg/contents:0': image_data})
            features[idx, :] = np.squeeze(predictions)

    return features

# Extract features
features = extract_features(image_list)

# Create array of filenames without .jpeg to join with ground-truth classification
keys = [os.path.splitext(os.path.basename(image))[0] for image in image_list]

# Save to disk
np.save('features', features)
np.save('keys', keys)
