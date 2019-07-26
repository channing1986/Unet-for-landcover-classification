__author__ = 'Changlin'
__version__ = 0.1

import numpy as np
from segmentation_models import UnetRegressor,Unet,pspnet#PSPNet
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import *
from keras.layers import Input
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.metrics import binary_accuracy,binary_crossentropy,categorical_accuracy
from tqdm import tqdm
import json
import tensorflow as tf
import params

from keras import metrics
from dataFunctions import my_class_weighted_loss

def get_model(net_name,num_class,weight_path,input_shape=[],weighted_loss=False):
    number_class=num_class
    if net_name=='psp':
        model_name='pspnet101_cityscapes'
        input_shape=(473,473,3)
        model = pspnet.PSPNet101(nb_classes=num_class, input_shape=input_shape,
                                    weights=model_name)
        model=model.model
    elif net_name=='psp_50':
        input_shape=(473,473,3)
        model_name='pspnet50_ade20k'
        #output_mode='sigmoid'
        model = pspnet.PSPNet50(nb_classes=num_class, input_shape=input_shape,
                                    weights=model_name)
        model=model.model

    elif net_name[-1:]=='c':
        if net_name=='unet_rgbh_c' or net_name=='unet_rgbc_c':
            if len(input_shape)<3:
                input_shape = [512,512,4]
        elif net_name=='unet_rgb_c':
            if len(input_shape)<3:
                input_shape = [512,512,3]
        elif net_name=='unet_msi_c':
            if len(input_shape)<3:
                input_shape = [512,512,3]
        elif net_name=='unet_msih_c' or net_name=='unet_msic_c':
            if len(input_shape)<3:
                input_shape = [512,512,9]
        from keras.layers import Input
        input_tensor = Input(shape=(input_shape[0],input_shape[1],input_shape[2]))
        model = Unet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=params.BACKBONE, 
                            encoder_weights=None, classes=num_class)
    if weighted_loss:
        loss=my_class_weighted_loss
    else:
        loss=params.SEMANTIC_LOSS
    lr=params.LEARN_RATE
    optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    if (len(weight_path)>2):
        model.load_weights(weight_path,True)
        print('use pre-trained weights',weight_path)
    model.compile(optimizer, loss=loss,metrics=[categorical_accuracy])

    model.summary()
    return model, input_shape