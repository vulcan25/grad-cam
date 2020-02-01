from keras.applications.vgg16 import preprocess_input
#from keras.preprocessing import image
from .keras_preprocessing_utils import load_img
import numpy as np

def image_to_arr(path, shape):
    img = load_img(path, target_size=shape[0:2])
    x = image.img_to_array(img)
    x = preprocess_input(x)
    return x
