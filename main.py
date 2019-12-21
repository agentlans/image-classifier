print('Starting program...')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Softmax, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing import image
from PIL import Image

from functools import reduce
import operator
import more_itertools as mi

import tqdm

import numpy as np

import os
import glob
import math

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("training", help="directory containing images for training classifier. Images must be organized in subdirectories for each class.")
parser.add_argument("unknown", help="directory containing images to classify. Subdirectories for each class will be automatically generated here.")
args = parser.parse_args()

# Parameters
batch_size = 32
train_dir = args.training
unknown_dir = args.unknown

# Loaders for training set
train_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
  train_dir,
  target_size=(224, 224),
  batch_size=batch_size)

sample_classes = list(train_generator.class_indices.keys())

# Loaders for unknown input data
unknown_files = glob.glob(unknown_dir + '/*.*')

def load_image_array(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  return image.img_to_array(img)

def make_input(filenames):
  x = np.stack(list(map(load_image_array, filenames)))
  return preprocess_input(x) # This is preprocessing for feature extractor

# Load feature extraction model
extractor = MobileNetV2(input_shape=(224,224,3), include_top=False)
extractor_output_shape = extractor.output_shape

# Fraction of neurons to keep so that we have 5 * batch_size number of inputs to dense layer
keep_frac = 5 * batch_size / reduce(operator.mul, extractor.output_shape[1:])

# Create model (a pretrained extractor + dense layer)
model = Sequential([
  extractor,
  Flatten(),
  Dropout(keep_frac),
  Dense(10, activation='relu'),
  BatchNormalization(),
  Dense(len(sample_classes)),
  Softmax()
])

# Freeze the feature extraction layers
model.layers[0].trainable = False

model.compile(optimizer='sgd',
  loss='categorical_crossentropy',
  metrics=['accuracy'])

stop_cb = EarlyStopping(monitor='loss', patience=2)

# Train model
model.fit_generator(
  train_generator,
  steps_per_epoch=5,
  #validation_data=test_generator,
  #validation_steps=2,
  epochs=20, callbacks=[stop_cb])

# Predict on unclassified images
print('Classifying the new files...')
total_batches = math.ceil(len(unknown_files) / batch_size)
unknown_generator = map(make_input, mi.chunked(unknown_files, batch_size))
pred = np.vstack(list(map(lambda x: model.predict(x), tqdm.tqdm(unknown_generator, total=total_batches))))
predicted_classes = pred.argmax(axis=1)

# Create directories in unknown_dir for each class
for img_class in list(train_generator.class_indices.keys()):
  unknown_subdir = os.path.join(unknown_dir, img_class)
  os.makedirs(unknown_subdir, exist_ok=True)

# Split filename into parts
def split_filename(filename):
  temp = filename.split('/')
  return reduce(operator.add, [x.split('\\') for x in temp])

# Move the images to their new classes
print('Moving files to new directories...')
class_names = dict([(y,x) for (x,y) in train_generator.class_indices.items()])
for original, new_class_id in tqdm.tqdm(zip(unknown_files, predicted_classes), total=len(unknown_files)):
  new_class = class_names[new_class_id]
  new_name = os.path.join(unknown_dir, new_class, split_filename(original)[-1])
  try:
    os.rename(original, new_name)
  except:
    print("Couldn't move " + original)


