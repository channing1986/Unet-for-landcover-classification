import numpy as np

from segmentation_models import UnetRegressor,Unet,pspnet#PSPNet
from keras.layers import Input

input_shape=(112,112,5)
input_tensor = Input(shape=(112,112,5))
model = Unet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=params.BACKBONE, 
                    encoder_weights=None, classes=number_class)