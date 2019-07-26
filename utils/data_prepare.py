
import json
from keras import backend as K
from keras.applications import imagenet_utils
from keras.layers import Dense,Input,merge,Flatten,Dropout,LSTM
from keras.models import Sequential,Model
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from functools import partial
import tensorflow as tf
import os
import cv2
Dataset_path='G:/DataSet/BSR/BSDS500/data/'
Img_Train='images/train/'
Img_Val='images/val/'
Img_Test='images/test/'

Gt_Train='groundTruth/train/'
Gt_Val='groundTruth/val/'
Gt_Test='groundTruth/test/'
def get_batch_inds(batch_size, idx, N):
    """
    Generates an array of indices of length N
    :param batch_size: the size of training batches
    :param idx: data to split into batches
    :param N: Maximum size
    :return batchInds: list of arrays of data of length batch_size
    """
    batchInds = []
    idx0 = 0

    toProcess = True
    while toProcess:
        idx1 = idx0 + batch_size
        if idx1 > N:
            idx1 = N
            idx0 = idx1 - batch_size
            toProcess = False
        batchInds.append(idx[idx0:idx1])
        idx0 = idx1

    return batchInds

def load_all_image(purpose, dset=Dataset_path, image_data_format='jpg'):
    import os
    from os import walk
    import string
    imgs = []
    gts=[]
    if purpose=='train':
        data_folder=Dataset_path+Img_Train
        gt_folder=Dataset_path+Gt_Train
    if purpose=='val':
        data_folder=Dataset_path+Img_Val
        gt_folder=Dataset_path+Gt_Val
    if purpose=='test':
        data_folder=Dataset_path+Img_Test
        gt_folder=Dataset_path+Gt_Test
    for (dirpath, dirnames, filenames) in walk(data_folder):
        for img in filenames:
            img_format=img[-3:]
            if img_format==image_data_format:
               imgs.append(data_folder+img)
               gts.append(gt_folder+img[:-3]+'tif')
        
    return imgs, gts

def load_all_facade_image(image_data_format='jpg'):
    import os
    from os import walk
    import string
    imgs = []
    gts=[]
    Dataset_path='G:/DataSet/BuildingFacade/etrims/etrims-db_v1/'
    Img_Train='images/04_etrims-ds/'
    Gt_Train='annotations-object/edge_map/'

    data_folder=Dataset_path+Img_Train
    gt_folder=Dataset_path+Gt_Train
 
    for (dirpath, dirnames, filenames) in walk(data_folder):
        for img in filenames:
            img_format=img[-3:]
            if img_format==image_data_format:
               imgs.append(data_folder+img)
               gts.append(gt_folder+img[:-4]+'_.png')
        
    return imgs, gts
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
    
def input_generator_mp(img_pathes,label_pathes,batch_size,task='segment',prediction=False,ignor_last=False):
    """

    """
    import multiprocessing as mp
    pool=mp.Pool();
    
    N = len(img_pathes) #total number of images

    idx = np.random.permutation(N) #shuffle the order
    if prediction:
        idx=range(0,N)
    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

            
    while True:
        for inds in batchInds:
            imgdata=[]
            labels=[]
            img_batch = [img_pathes[ind] for ind in inds]
            label_batch = [label_pathes[ind] for ind in inds]
            img_label=zip(img_batch,label_batch)
            #res=load_img_label(img_batch[0],label_batch[0])######
            if task=='segment':
                res=pool.starmap(load_img_label_seg,img_label)
            elif task=='edge':
                res=pool.starmap(load_img_label,img_label)
            for img,lable in res:
                imgdata.append(img)
                labels.append(lable)

            labels = np.array(labels, np.float32)
            if ignor_last:
                labels=labels[:,:,:,:-1]
            imgdata = np.array(imgdata, np.float32)
    #return imgdata,labels
            yield (imgdata, labels)

def load_img_label_seg(rgb_file,gts_file):
    rgb = cv2.imread(rgb_file)
 #   rgbh = image.img_to_array(rgbh)
    gts = cv2.imread(gts_file,0)
    
    #rgb=cv2.resize(rgb,(320,320))
    rgb=np.array(rgb,np.float32)
    rgb = preprocess_input(rgb,mode='tf')

    gts[np.where(gts == 0)] = 5
    gts = np.expand_dims(gts, axis=-1)
    #gts = Image.open(gts_file )
    #gts = img_to_array(gts,data_format='channels_last').astype(int)
    #gts=np.squeeze(gts)
    #np.savetxt('gts1.txt', gts)


    return rgb, gts


def input_generator_mp_p(img_pathes,label_pathes,batch_size):
    """

    """
    import multiprocessing as mp
    pool=mp.Pool();
    
    N = len(img_pathes) #total number of images

    idx = np.random.permutation(N) #shuffle the order
    idx=range(0,N)
    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

            
    while True:
        for inds in batchInds:
            imgdata=[]
            labels=[]
            img_batch = [img_pathes[ind] for ind in inds]
            label_batch = [label_pathes[ind] for ind in inds]
            img_label=zip(img_batch,label_batch)
            res=pool.starmap(load_img_label,img_label)
            for img,lable in res:
                imgdata.append(img)
                labels.append(lable)
            labels = np.array(labels, np.float32)
            imgdata = np.array(imgdata, np.float32)/255.
    #return imgdata,labels
            yield (imgdata, labels)
def load_img_label(rgb_file,gts_file):

    #rgb_file = inputDict['rgb']
    #gts_file = inputDict['gts']
    rgb = cv2.imread(rgb_file)
    rgb=np.array(rgb,np.float32)
    rgb = preprocess_input(rgb,mode='tf')
 #   rgbh = image.img_to_array(rgbh)
    gts = cv2.imread(gts_file,0)
    gts[np.where(gts == 0)] = 5
    # rgb=cv2.resize(rgb,(320,320))
    # gts=cv2.resize(gts,(320,320))

    y = np.zeros((gts.shape[0], gts.shape[1],5), dtype=np.float32)
    for i in range(gts.shape[0]):
        for j in range(gts.shape[1]):

            # if gts[i][j]<1:     ## for 2 channels output
            #     cc=0
            # else:
            #     cc=1
            cc=gts[i,j]-1
            y[i,j,cc]=1


    return rgb, y
def input_generator(img_pathes,label_pathes,batch_size):
    """
    """

    N = len(img_pathes) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    executor = ProcessPoolExecutor(max_workers=1)

    while True:
        for inds in batchInds:
            img_batch = [img_pathes[ind] for ind in inds]
            label_batch = [label_pathes[ind] for ind in inds]

            imgdata, gts= load_cnn_batch(img_batch, label_batch,  executor)    
            #imgdata, gts= load_cnn_batch_mp(img_batch, label_batch) 
            yield (imgdata, gts)            



def load_cnn_batch(img_batch, label_batch,  executor):
    """
    """
    results=[]
    imgdata=[]
    labels=[]
    futures = []
    for i in range(0, len(img_batch)):
        currInput = {}
        currInput['gts'] =label_batch[i]
        currInput['rgb'] = img_batch[i]

        futures.append(executor.submit(_load_batch_helper, currInput))
        results = [future.result() for future in futures]
        #results.append( _load_batch_helper(currInput))
    for  i, result in enumerate(results):
        imgdata.append(result[0])
        labels.append(result[1])

    #imgdata = imagenet_utils.preprocess_input(imgdata)
    #imgdata = imgdata / 255.0
    y_train = np.array(labels, np.float32)
    #y_train=labels
    #y_train=labels
    x_train = np.array(imgdata, np.float32)/255.
    return x_train, y_train

def _load_batch_helper(inputDict):
    """
    Helper for load_cnn_batch that actually loads imagery and supports parallel processing
    :param inputDict: dict containing the data and metadataStats that will be used to load imagery
    :return currOutput: dict with image data, metadata, and the associated label
    """
    #print("fsf")
    #return
    rgb_file = inputDict['rgb']
    gts_file = inputDict['gts']
    rgb = cv2.imread(rgb_file)
 #   rgbh = image.img_to_array(rgbh)
    gts = cv2.imread(gts_file,0)
    rgb=cv2.resize(rgb,(320,320))
    gts=cv2.resize(gts,(320,320))

    y = np.zeros((gts.shape[0], gts.shape[1],2), dtype=np.float32)
    for i in range(gts.shape[0]):
        for j in range(gts.shape[1]):

            if gts[i][j]<1:     ## for 2 channels output
                cc=0
            else:
                cc=1
            y[i,j,cc]=1

            #if gts[i][j]<1:          ## for 1 channel output
            #    y[i][j]=0
            #else:
            #     y[i][j]=1
#    return y
    #y = image.img_to_array(gts)
    #currOutput = {}
    #currOutput['imgs'] = rgbh
    #currOutput['labels'] = y
    #y_=y*255
    #gts_=gts*255;
    ##cv2.imshow('ss1',y_)
    ##cv2.imshow('ss2',gts_)
    #np.savetxt('ss1.txt',y)
    #np.savetxt('ss2.txt',gts)
    #cv2.imshow('ss3',y[:,:,0])
    #cv2.imshow('ss4',y[:,:,1])
    #cv2.waitKey(0)
    return rgb, y