import tensorflow as tf
import numpy as np
import tensorflow
import pandas as pd
import yaml
from IPython import embed
from model import get_pose_net
import os
import json
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import keras


from preprocessing import KeyPointsDataset, reading_annot_data
from loss import huber_loss

with open("config.yaml") as f:
	params = yaml.safe_load(f)


IMG_DIR = "/home/bhatt/shubham/LPN_Implementation/train2014/"


IMG_SIZE = params['img_width']

input_dir = params["input_dir"]
BATCH_SIZE = params["batch_size"]
NUM_KEYPOINTS =17 * 2
 

all_data = reading_annot_data(params["json_file"])

samples = list(all_data.keys())
np.random.shuffle(samples)
train_keys, validation_keys = (
    samples[int(len(samples) * 0.15) :],
    samples[: int(len(samples) * 0.15)],
)


train_aug = iaa.Sequential(
    [
        iaa.Resize(IMG_SIZE, interpolation="linear")
        # iaa.Fliplr(0.3),
        # # `Sometimes()` applies a function randomly to the inputs with
        # # a given probability (0.3, in this case).
        # iaa.Sometimes(0.3, iaa.Affine(rotate=10, scale=(0.5, 0.7))),
    ]
)

test_aug = iaa.Sequential([iaa.Resize(IMG_SIZE, interpolation="linear")])


train_dataset = KeyPointsDataset(train_keys, train_aug,batch_size=BATCH_SIZE, all_data=all_data)
validation_dataset = KeyPointsDataset(validation_keys, test_aug,batch_size=BATCH_SIZE,train=False, all_data=all_data)

model = get_pose_net(params, is_train = True)

model.summary()
model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(1e-4))
model.fit(train_dataset, validation_data=validation_dataset, epochs=2)





