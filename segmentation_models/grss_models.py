__author__ = 'Changlin'
__version__ = 0.1

import numpy as np

from segmentation_models import UnetRegressor,Unet,pspnet#PSPNet
#from segmentation_models import psp_50
from dataFunctions import *
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import *
from keras.layers import Input
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam, SGD
from tqdm import tqdm
import json
import tensorflow as tf
import params

from model import models
from track1_data_height import no_nan_mse,no_nan_mse_evenloss,no_nan_mse_evenloss_bridge
from track1_data_height import load_all_data_files,input_load_train_data,input_generator_online_process,input_generator_RGB,load_all_data_files_balanced_patches
from dataFunctions import my_weighted_loss_5_classes,my_weighted_loss,my_weighted_loss_3_classes,my_tf_balanced_loss

def get_model(net_name,num_class,weight_path,input_shape=[]):
    from segmentation_models import pspnet#PSPNet
    number_class=num_class

    if net_name=='psp':
        model_name='pspnet101_cityscapes'
        input_shape=(473,473,3)
        model = pspnet.PSPNet101(nb_classes=number_class, input_shape=input_shape,
                                    weights=model_name)
        model=model.model
    elif net_name=='psp_50':
        input_shape=(473,473,3)
        model_name='pspnet50_ade20k'
        #output_mode='sigmoid'
        model = pspnet.PSPNet50(nb_classes=number_class, input_shape=input_shape,
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
                            encoder_weights=None, classes=number_class)
    elif net_name[-1:]=='h':
        if net_name=='unet_rgbh_h' or net_name=='unet_rgbc_h':
            if len(input_shape)<3:
                input_shape = [512,512,4]
        elif net_name=='unet_rgb_h':
            if len(input_shape)<3:
                input_shape = [512,512,3]
        elif net_name=='unet_msi_h' or net_name=='unet_msi_c':
            if len(input_shape)<3:
                input_shape = [512,512,8]
        elif net_name=='unet_msih_h' or net_name=='unet_msic_h':
            if len(input_shape)<3:
                input_shape = [512,512,9]
        from keras.layers import Input
        input_tensor = Input(shape=(input_shape[0],input_shape[1],input_shape[2]))
        model = UnetRegressor(input_shape=input_shape, input_tensor=input_tensor, 
                    backbone_name=params.BACKBONE)

    if net_name[-1:]=='h':
        loss=no_nan_mse_evenloss
    elif number_class==2:
        loss=my_weighted_loss
    elif number_class==5:
        loss=my_weighted_loss_5_classes
    elif number_class==3:
            loss=my_weighted_loss_3_classes
            #loss='categorical_crossentropy'
            #loss=my_tf_balanced_loss
    optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    if (len(weight_path)>2):
        model.load_weights(weight_path)
        print('use pre-trained weights',weight_path)
    model.compile(optimizer, loss=loss)

    model.summary()
    return model, input_shape