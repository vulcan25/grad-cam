#PARAMS = './example/model/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
#MODEL_SOURCE_PATH='./example/src/vgg16.py'
#MODEL_SOURCE_DEFINITION='vgg16'

PARAMS  ='/model_data/yolo_weights.h5'
MODEL_SOURCE_PATH='/code/yolo3/model.py'
MODEL_SOURCE_DEFINITION='yolo_body'

LAYERS = ['block5_conv3','block4_conv3']
ARGS = [
        [224,224], # image size
        3, # channel
        1000 # classes
       ]
import cv2
import importlib.machinery as im_machinery
import types
import keras_pkg.grad_cam as k_grad_cam
import keras_pkg.util as k_util
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
from io import BytesIO

def __function_loader(mod_path, func_name, mod_name='mod'):
    loader = im_machinery.SourceFileLoader(mod_name, mod_path)
    mod = types.ModuleType(loader.name)
    # import module
    loader.exec_module(mod)
    # Check whether module is defined in function
    assert func_name in dir(mod), '{0} is undefined in {1}'.format(func_name, mod_path)
    func_str = mod_name + '.' + func_name

    return eval(func_str)

def __keras_grad_cam(input_image):
    """
    model_definition = __function_loader(
        MODEL_SOURCE_PATH,
        MODEL_SOURCE_DEFINITION)
    # get model from source code
    model = k_util.get_model(
        PARAMS,
        model_definition(*ARGS)
        )
    """
    model = k_util.get_model(PARAMS)
    model.summary()

    def image_to_arr(path, shape):
        img = image.load_img(path)#, target_size=shape[0:2])
        x = image.img_to_array(img)
        x = preprocess_input(x)
        return x

    #preprocessing_func = __function_loader(
     #           IMAGE_SOURCE_PATH,
      #          IMAGE_SOURCE_DEFINITION
       #         )


    k_util.show_predicted_class(model, [input_image], image_to_arr)
    
    results = k_grad_cam.exec(
                model,
                LAYERS,
                [input_image],
                image_to_arr)

    return results, model.name

def keras_grad_cam(input_image):
    results, model_name = __keras_grad_cam(input_image)

    print ('+==========================+')
    print ('+==========================+')
    outputs = []

    for (layer_idx, layer_results) in enumerate(results):
        for (cam_idx, (cam, heatmap)) in enumerate(layer_results): 
            
            d = {}

            d['model_name'] = model_name
            d['layer'] = LAYERS[layer_idx]
            
            # cv2.imwrite(os.path.join(config.image.output, output_name), cam

            is_success, output_stream = cv2.imencode(".jpg", cam)
            io_buff = BytesIO(output_stream)
            
            d['file'] = io_buff.read()
            
            d['cam'] = cam
            d['heatmap'] = heatmap

            outputs.append(d)

    return outputs
            
if __name__ == '__main__':
    import os
    
    with open('hydrant.jpg', 'rb') as f:
        res = keras_grad_cam(f)

    for d in res:
        with open(os.path.join(d['model_name'] + '-' + d['layer'] +  '.png'),'wb') as f:
            f.write(d['file'])

    
