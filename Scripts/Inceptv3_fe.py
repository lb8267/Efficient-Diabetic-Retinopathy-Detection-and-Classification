import os
import glob

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np

model_dir = '/content/drive/MyDrive/inception-2015-12-05'

images_dir = '/content/drive/MyDrive/Disease_Grading/original_Images/train-resized-256/'
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
    features1 = np.empty((len(image_list), nb_features))

    create_graph()

    with tf.compat.v1.Session() as sess:
        bottleneck_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        print('Processing %i images' % len(image_list))
        for idx, image in enumerate(image_list):
            
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(bottleneck_tensor, {'DecodeJpeg/contents:0': image_data})
            features1[idx, :] = np.squeeze(predictions)

    return features1

# Extract features
features1 = extract_features(image_list)

# Create array of filenames without .jpeg to join with ground-truth classification
keys1 = [os.path.splitext(os.path.basename(image))[0] for image in image_list]

# Save to disk
np.save('features1', features1)
np.save('keys1', keys1)


keys1 = np.load('keys1.npy')
features1 = np.load('features1.npy')
labels_df = pd.read_csv('/content/drive/MyDrive/idrid_Disease_Grading/Groundtruth_labels/trainLabels_master_256_2columns.csv',index_col="train_image_name")

# Check lengths are the same for all raw data
len(keys1), len(features1), len(labels_df)

labels_df.head(4)

features_df.head(4)

features_df = pd.DataFrame(features1, index=keys1)
df = pd.merge(features_df, labels_df, how='inner', left_index=True, right_index=True)

df.head(4)

print(df)

df.level.hist(figsize=(4,3));
df.level.value_counts()

min_class = df.level.value_counts().min() # number of samples in smallest class
min_class

feature_cols = df.columns[:-1]  # all columns except 'level'
X = df[feature_cols].values
y = df['level'].values

X.shape, y.shape

