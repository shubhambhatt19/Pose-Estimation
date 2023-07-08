import numpy as np
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
import imgaug.augmenters as iaa
import keras
import yaml
import json
import matplotlib.pyplot as plt
import os


with open("config.yaml") as f:
    params = yaml.safe_load(f)


IMG_SIZE = params["img_height"]
NUM_KEYPOINTS = params["MODEL"]["NUM_JOINTS"]
IMG_DIR = params["input_dir"]
# loading data using generator: 
# https://keras.io/examples/vision/keypoint_detection/

class KeyPointsDataset(keras.utils.Sequence):
    def __init__(self, image_keys, aug, batch_size=None, train=True, all_data=None):
        self.image_keys = image_keys
        self.aug = aug
        self.batch_size = batch_size
        self.train = train
        self.all_data = all_data
        self.on_epoch_end()

    def __len__(self):
        return 100 #    len(self.image_keys) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_keys))
        if self.train:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        image_keys_temp = [self.image_keys[k] for k in indexes]
        (images, keypoints) = self.__data_generation(image_keys_temp)

        return (images, keypoints)

    def __data_generation(self, image_keys_temp):
        batch_images = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 3), dtype="int")
        batch_keypoints = np.empty(
            (self.batch_size, 1, 1, NUM_KEYPOINTS), dtype="float32"
        )

        for i, key in enumerate(image_keys_temp):
            try:
            	data = get_data(self.all_data,key)
            except FileNotFoundError:
            	continue
            current_keypoint = np.array(data["joints"])[:, :2]
            kps = []

            # To apply our data augmentation pipeline, we first need to
            # form Keypoint objects with the original coordinates.
            for j in range(0, len(current_keypoint)):
                kps.append(Keypoint(x=current_keypoint[j][0], y=current_keypoint[j][1]))

            # We then project the original image and its keypoint coordinates.
            current_image = data["img_data"]
            kps_obj = KeypointsOnImage(kps, shape=current_image.shape)

            # Apply the augmentation pipeline.
            (new_image, new_kps_obj) = self.aug(image=current_image, keypoints=kps_obj)
            if len(new_image.shape) <= 2:
            	continue
            batch_images[i,] = new_image

            # Parse the coordinates from the new keypoint object.
            kp_temp = []
            for keypoint in new_kps_obj:
                kp_temp.append(np.nan_to_num(keypoint.x))
                kp_temp.append(np.nan_to_num(keypoint.y))

            # More on why this reshaping later.
            batch_keypoints[i,] = np.array(kp_temp).reshape(1, 1, 17*2) #np.array(kp_temp).reshape(1, 1, 24 * 2)

        # Scale the coordinates to [0, 1] range.
        batch_keypoints = batch_keypoints / IMG_SIZE
        return (batch_images, batch_keypoints)


# data reading start
def reading_annot_data(json_file):
    data = {}
    targets = []
    filenames = []

    with open(json_file) as f:
        temp_annot_data = json.load(f)

    final_annot_data = dict()
    for i in range(len(temp_annot_data["images"])):
        annot_data = dict()
        joints=[]
        for j in range(0,51,3):
            joints.append([temp_annot_data["annotations"][i]["keypoints"][j], temp_annot_data["annotations"][i]["keypoints"][j+1],temp_annot_data["annotations"][i]["keypoints"][j+2]])
        annot_data["joints"] = joints
        annot_data["category_id"] = temp_annot_data["annotations"][i]["category_id"]
        final_annot_data[temp_annot_data['images'][i]['file_name']] = annot_data
    return final_annot_data





def get_data(all_data, name):
    data = all_data[name]
    
    img_data = plt.imread(os.path.join(IMG_DIR, name))
    # If the image is RGBA convert it to RGB.
    if img_data.shape[-1] == 4:
        img_data = img_data.astype(np.uint8)
        img_data = Image.fromarray(img_data)
        img_data = np.array(img_data.convert("RGB"))
    data["img_data"] = img_data
    return data

