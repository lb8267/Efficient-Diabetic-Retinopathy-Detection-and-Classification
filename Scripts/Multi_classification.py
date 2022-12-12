

import tensorflow as tf
#import keras
from tensorflow import lite

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from IPython.display import clear_output
from tensorflow import keras
import time
import datetime

import shutil

from tensorflow.compat.v1 import ConfigProto,InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

train_datagen = ImageDataGenerator(rescale=1./255)

x_train = train_datagen.flow_from_directory(
    directory=r'/content/drive/MyDrive/Train_Classification_images/',
    batch_size=32,
    target_size=(224,224),
    class_mode="categorical",
    shuffle=True,
    seed=42
)
train_generator = x_train
class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(train_generator.classes), y = train_generator.classes)
class_weights = dict(enumerate(class_weights))
print("train_generator class_weights =",  class_weights)

validation_datagen = ImageDataGenerator(rescale=1./255)

x_validation = validation_datagen.flow_from_directory(
    directory=r'/content/drive/MyDrive/Test_classification_images/',
    batch_size=32,
    target_size=(224,224),
    class_mode="categorical",
    shuffle=True,
    seed=42
)
validation_generator = x_validation
class_weights_val = class_weight.compute_class_weight('balanced', classes = np.unique(validation_generator.classes), y = validation_generator.classes)
class_weights_val = dict(enumerate(class_weights_val))
class_weights_val
print("val_generator class_weights =",  class_weights_val)

test_datagen = ImageDataGenerator(rescale=1./255)

x_test = test_datagen.flow_from_directory(
    directory=r'/content/drive/MyDrive/Test_classification_images/',
    batch_size=32,
    target_size=(224,224),
    class_mode="categorical",
    shuffle=False,
    seed=42
)

test_generator = x_test

num_train_samples = train_generator.n
num_val_samples = validation_generator.n
train_batch_size = 256
val_batch_size = 256


train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)
print(train_steps)
print(val_steps)

p = x_train.next()
print((p[0][0]).shape)
(plt.imshow(p[0][0][:,:,:]) )

lr =0.0001
def create_dcnnmodel(n_classes=5,learning_rate=lr):
    full_model = Sequential()
    full_model.add(raw_model)
    full_model.add(GlobalAveragePooling2D())
    full_model.add(Dropout(0.2))
    full_model.add(Dense(512,activation='relu'))
    full_model.add(Dropout(0.1))
    full_model.add(Dense(5, activation = 'softmax'))
    full_model.compile(Adam(lr=learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    full_model.summary()
    return full_model


keras.backend.clear_session()

model = create_dcnnmodel(n_classes=5,learning_rate=lr)


history = model.fit_generator(train_generator, steps_per_epoch=train_steps, 
                            validation_data=validation_generator,
                            validation_steps=val_steps,
                            epochs=100, verbose=1
                            
                            )

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

