PARAMS = './example/model/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
INPUT_IMAGE = './example/images/hydrant.jpg'
MODEL_SOURCE_PATH='./example/src/vgg16.py'
MODEL_SOURCE_DEFINITION='vgg16'
IMAGE_PATH='./examples/images'
LAYERS = ['block5_conv3','block4_conv3']

def __function_loader(mod_path, func_name, mod_name='mod'):
    loader = im_machinery.SourceFileLoader(mod_name, mod_path)
    mod = types.ModuleType(loader.name)
    # import module
    loader.exec_module(mod)
    # Check whether module is defined in function
    assert func_name in dir(mod), '{0} is undefined in {1}'.format(func_name, mod_path)
    func_str = mod_name + '.' + func_name

    return eval(func_str)

import cv2
import importlib.machinery as im_machinery
import types
import keras_pkg.grad_cam as k_grad_cam
import keras_pkg.util as k_util

#model = k_util.get_model(PARAMS)


model_definition = __function_loader(
    MODEL_SOURCE_PATH,
    MODEL_SOURCE_DEFINITION)
# get model from source code
model = k_util.get_model(
    PARAMS,
    model_definition([224,224],3,1000))


model.summary()

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np

#from keras_preprocessing_utils import load_img

def image_to_arr(path, shape):
    img = image.load_img(path, target_size=shape[0:2])
    x = image.img_to_array(img)
    x = preprocess_input(x)
    return x

#preprocessing_func = __function_loader(
 #           IMAGE_SOURCE_PATH,
  #          IMAGE_SOURCE_DEFINITION
   #         )

with open (INPUT_IMAGE,'rb') as i:
    IMAGE = i

    k_util.show_predicted_class(model, [IMAGE], image_to_arr)

    results = k_grad_cam.exec(
            model,
            LAYERS,
            [IMAGE],
            image_to_arr)

print ('=================')

print (results,  model.name)
