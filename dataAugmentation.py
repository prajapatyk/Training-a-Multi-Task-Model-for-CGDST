# %%
import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import shutil

# %%
#takes image path as input and return an numpy array
def convertImageToNumpyArray(imgPath):
    img = cv2.imread(imgPath)
    return np.array(img)

# %%
#takes the image as numpy array
#returns the results as numpy array
seed = (2,7)

def random_brightness(img , max_delta):
    aug_img = tf.image.stateless_random_brightness(img, max_delta, seed).numpy()
    return aug_img


def random_contrast(img, lower, upper):
    aug_img = tf.image.stateless_random_contrast(img, lower, upper, seed).numpy()
    return aug_img


def random_hue(img, max_delta):
    aug_img = tf.image.stateless_random_hue(img, max_delta, seed).numpy()
    return aug_img


def random_saturation(img, lower, upper):
    aug_img = tf.image.stateless_random_saturation(img, lower, upper, seed).numpy()
    return aug_img


def add_saltAndPepper_noise(img, salt_vs_pepper, amount):
    noise_img = random_noise(img, mode = 's&p', salt_vs_pepper = salt_vs_pepper, amount = amount, seed = seed)
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img


def add_gaussian_noise(img, mean, var):
    noise_img = random_noise(img, mode = 'gaussian', mean = mean, var = var, seed = seed)
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img


def add_speckle_noise(img, mean, var):
    noise_img = random_noise(img, mode = 'speckle', mean = mean, var = var, seed = seed)
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img

# %%
def get_augmented_images(directoryPath, imgFile):
    imgNpArray = convertImageToNumpyArray(os.path.join(directoryPath,imgFile))
    augImages = []
    
    augImages.append(random_brightness(imgNpArray,0.5))
    augImages.append(random_contrast(imgNpArray,0.2,1))
    augImages.append(add_saltAndPepper_noise(imgNpArray,0.5,0.08))
    augImages.append(add_gaussian_noise(imgNpArray,0,0.04))
    
    return augImages

#random brightness: 0.5
#random contrast: [0.2, 0.8]
#salt and pepper noise: [0.02, 0.08]
#guassian noise:meann &var : [(0,0.04)]
#speckle noise: [(0,0.04)]



