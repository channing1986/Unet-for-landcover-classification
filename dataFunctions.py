__author__ = 'jhuapl'
__version__ = 0.1

import os
import numpy as np
from glob import glob
import params
import tifffile
import cv2
from keras.applications import imagenet_utils
from keras.utils import to_categorical
from matplotlib import pyplot as plt

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    RandomBrightnessContrast
)


def parse_args(argv, params):
    """
    Parses input argument to determine if the code trains or tests a model and whether or not 
        the task is semantic segmentation or single-view depth prediction.
    :param argv: input arguments from main
    :param params: input parameters from params.py
    :return: modes of operation
    """
    
    isTrain = None
    
    argOptions1 = '1st argument options: train, test.'
    argOptions2 = '2nd argument options: semantic, single-view.'
    noArgStr = 'No arguments provided.' 
    incorrectArgStr = 'Incorrect argument provided.'
    insufficientArgStr = 'Not enough arguments provided.'
    exampleUsageStr = 'python runBaseline.py train semantic'
    
    try:
        trainStr = argv[1].lower()
    except:
        raise ValueError('%s %s %s' % (noArgStr,argOptions1,exampleUsageStr))
    
    try:
        modeStr = argv[2].lower()
    except:
        raise ValueError('%s %s %s' % (insufficientArgStr,argOptions2,exampleUsageStr))
        
    if trainStr == 'train':
        isTrain = True
    elif trainStr == 'test':
        isTrain = False
    else:
        raise ValueError('%s %s %s' % (incorrectArgStr,argOptions1,exampleUsageStr))
 
    if modeStr == 'semantic':
        mode = params.SEMANTIC_MODE
    elif modeStr == 'single-view':
        mode = params.SINGLEVIEW_MODE
    else:
        raise ValueError('%s %s %s' % (incorrectArgStr,argOptions2,exampleUsageStr))
        
    if (mode==params.SEMANTIC_MODE) and (params.NUM_CATEGORIES==1) and (params.SEMANTIC_LOSS=='categorical_crossentropy'):
        _=input('Warning: NUM_CATEGORIES is 1, but loss is not binary_crossentropy. You should probably change this in params.py, but press enter to continue.')
        
    return isTrain,mode


def get_image_paths_bb(params, isTest=None):
    """
    Generates a list semantic ground truth files, which are used to load RGB and 
        depth files later with string replacements (i.e., only use image data that has semantic ground truth)
    :param params: input parameters from params.py
    :param isTest: determines whether or not to get image files for training or testing
    :return: list of paths to use for training
    """
    
    if isTest:
        return glob(os.path.join(params.TEST_DIR, '*%s*.%s' % (params.IMG_FILE_STR,params.IMG_FILE_EXT)))
    else:
        img_paths = []
        wildcard_image = '*%s.%s' % (params.CLASS_FILE_STR, params.LABEL_FILE_EXT)
        glob_path = os.path.join(params.LABEL_DIR, wildcard_image)
        curr_paths = glob(glob_path)
        for currPath in curr_paths:
            image_name = os.path.split(currPath)[-1]
            image_name = image_name.replace(params.CLASS_FILE_STR, params.IMG_FILE_STR)
            image_name = image_name.replace(params.LABEL_FILE_EXT, params.IMG_FILE_EXT)
            img_paths.append(os.path.join(params.TRAIN_DIR, image_name))
    return img_paths

def get_image_paths(params, isTest=None):
    """
    Generates a list semantic ground truth files, which are used to load RGB and 
        depth files later with string replacements (i.e., only use image data that has semantic ground truth)
    :param params: input parameters from params.py
    :param isTest: determines whether or not to get image files for training or testing
    :return: list of paths to use for training
    """
    
    if isTest:
        return glob(os.path.join(params.TEST_DIR, '*RGB.%s' % (params.IMG_FILE_EXT)))
    else:
        img_paths = []
        wildcard_image = '*.%s' % (params.LABEL_FILE_EXT)
        glob_path = os.path.join(params.LABEL_DIR, wildcard_image)
        curr_paths = glob(glob_path)
        for currPath in curr_paths:
            image_name = os.path.split(currPath)[-1]
            # image_name = image_name.replace(params.CLASS_FILE_STR, params.IMG_FILE_STR)
            # image_name = image_name.replace(params.LABEL_FILE_EXT, params.IMG_FILE_EXT)
            img_paths.append(os.path.join(params.TRAIN_DIR, image_name))
    return img_paths

def load_img(imgPath):
    """
    Load image
    :param imgPath: path of the image to load
    :return: numpy array of the image
    """
    if imgPath.endswith('.tif'):
        img = tifffile.imread(imgPath)
    else:
        #raise ValueError('Install pillow and uncomment line in load_img')
        img = np.array(cv2.imread(imgPath))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    image=image.astype('uint8')
    # mask=(mask*40).astype('uint8')
    # cv2.imshow('img',image)
    # cv2.imshow('mask',mask)
    #cv2.waitkey(1)
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))
        original_image=original_image.astype('uint8')
        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()
def image_augmentation(currImg, labelMask):
    """
    Apply random image augmentations
    :param currImg: original image
    :param labelMask: original ground truth
    :return: post-augmentation image and ground truth data
    """
    aug = Compose([VerticalFlip(p=0.5), #RandomBrightnessContrast(p=0.05),             
            RandomRotate90(p=0.5),HorizontalFlip(p=0.5),Transpose(p=0.5)])

    augmented = aug(image=currImg, mask=labelMask)
    imageMedium = augmented['image']
    labelMedium = augmented['mask']
    #visualize(imageMedium,labelMedium,currImg,labelMask)
    return imageMedium,labelMedium

def image_augmentation_test(currImg, labelMask):
    """
    Apply random image augmentations
    :param currImg: original image
    :param labelMask: original ground truth
    :return: post-augmentation image and ground truth data
    """
    #aug = Compose([RandomGamma(gamma_limit=(70, 130), p=0.6),Transpose(p=0.5),GridDistortion(distort_limit=0.2,p=0.6),HorizontalFlip, RandomRotate90(p=0.5),VerticalFlip(p=0.5),])
    aug = Compose([HorizontalFlip(p=0.5),GridDistortion(distort_limit=0.2, p=0.5)             
            ])

    augmented = aug(image=currImg, mask=labelMask)
    imageMedium = augmented['image']
    labelMedium = augmented['mask']
    #visualize(imageMedium,labelMedium,currImg,labelMask)
    return imageMedium,labelMedium


def image_batch_preprocess(imgBatch, params, meanVals):
    """
    Apply preprocessing operations to the image data that also need to be applied during inference
    :param imgBatch: numpy array containing image data
    :param params: input parameters from params.py
    :param meanVals: used for mean subtraction if non-rgb imagery
    :return: numpy array containing preprocessed image data
    """
    if params.NUM_CHANNELS==3:
        imgBatch  = imagenet_utils.preprocess_input(imgBatch)
        imgBatch = imgBatch / 255.0
    else:
        for c in range(params.NUM_CATEGORIES):
            imgBatch[:,:,:,c] -= meanVals[c]
        imgBatch = imgBatch / params.MAX_VAL
    return imgBatch


def get_label_mask(labelPath, params, mode):
    """
    Loads the ground truth image (semantic or depth)
    :param labelPath: Path to the ground truth file (CLS or AGL file)
    :param params: input parameters from params.py
    :param mode: either SEMANTIC_MODE or SINGLEVIEW_MODE from params (0 or 1)
    :return: numpy array containing ground truth
    """
    currLabel = load_img(labelPath)
    if mode == params.SINGLEVIEW_MODE:
        currLabel[np.isnan(currLabel)] = params.IGNORE_VALUE
    elif mode == params.SEMANTIC_MODE:
        if np.max(currLabel)>params.NUM_CATEGORIES:
            currLabel = convert_labels(currLabel, params, toLasStandard=False)
    return currLabel

def get_label_mask_bb(labelPath, params, mode):
    """
    Loads the ground truth image (semantic or depth)
    :param labelPath: Path to the ground truth file (CLS or AGL file)
    :param params: input parameters from params.py
    :param mode: either SEMANTIC_MODE or SINGLEVIEW_MODE from params (0 or 1)
    :return: numpy array containing ground truth
    """
    currLabel = load_img(labelPath)
    if mode == params.SINGLEVIEW_MODE:
        currLabel[np.isnan(currLabel)] = params.IGNORE_VALUE
    elif mode == params.SEMANTIC_MODE:
        if np.max(currLabel)>params.NUM_CATEGORIES:
            currLabel = convert_labels(currLabel, params, toLasStandard=False)
        if params.NUM_CATEGORIES > 1:
            currLabel = to_categorical(currLabel, num_classes=params.NUM_CATEGORIES+1)
    return currLabel

def load_batch_bb(inds, trainData, params, mode, meanVals=None):
    """
    Given the batch indices, load the images and ground truth (labels or depth data)
    :param inds: batch indices
    :param trainData: training paths of CLS files (string replacement to get RGB and depth files) and starting x,y pixel positions (can be      non-zero if blocking is set to happen in params.py)
    :param params: input parameters from params.py
    :param mode: either SEMANTIC_MODE or SINGLEVIEW_MODE from params (0 or 1)
    :param meanVals: used for mean subtraction if non-rgb imagery
    :return: numpy arrays for image and ground truth batch data
    """
    
    if params.BLOCK_IMAGES:
        batchShape = (params.BATCH_SZ, params.BLOCK_SZ[0], params.BLOCK_SZ[1])
    else:
        batchShape = (params.BATCH_SZ, params.IMG_SZ[0], params.IMG_SZ[1])
    
    imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS))
    
    numChannels = None
    if mode == params.SINGLEVIEW_MODE:
        numChannels = 1
        labelReplaceStr = params.DEPTH_FILE_STR
    elif mode == params.SEMANTIC_MODE:
        numChannels = params.NUM_CATEGORIES
        labelReplaceStr = params.CLASS_FILE_STR

    labelBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], numChannels))
    
    batchInd = 0
    for i in inds:
        currData = trainData[i]
        imgPath = currData[0]
        if params.LABEL_DIR != params.TRAIN_DIR:
            imageName = os.path.split(imgPath)[-1]
            if params.LABEL_FILE_EXT != params.IMG_FILE_EXT:
                imageName = imageName.replace('.'+params.IMG_FILE_EXT, '.'+params.LABEL_FILE_EXT)
            labelPath = os.path.join(params.LABEL_DIR, imageName.replace(params.IMG_FILE_STR, labelReplaceStr))
        else:
            labelPath = imgPath.replace(params.IMG_FILE_STR,labelReplaceStr)
        currImg = load_img(imgPath)
        currLabel = get_label_mask(labelPath, params, mode)
        ###make sure the input size is equal to designed size
        if currImg.shape[0]!=batchShape[1] or currImg.shape[1]!=batchShape[2]:
            currImg=cv2.resize(currImg,(batchShape[1],batchShape[2]))
            currLabel=cv2.resize(currLabel,(batchShape[1],batchShape[2]),cv2.INTER_NEAREST)


        rStart,cStart = currData[1:3]
        rEnd,cEnd = (rStart+batchShape[1],cStart+batchShape[2])
        currImg = currImg[rStart:rEnd, cStart:cEnd, :]
        if mode == params.SINGLEVIEW_MODE:
            currLabel = currLabel[rStart:rEnd, cStart:cEnd]
        else:
            currLabel = currLabel[rStart:rEnd, cStart:cEnd, :]
        
        imageMedium,labelMedium = image_augmentation(currImg, currLabel)

        imgBatch[batchInd,:,:,:] = imageMedium
        if mode == params.SINGLEVIEW_MODE:
            labelBatch[batchInd,:,:,0] = labelMedium
        else:
            labelBatch[batchInd,:,:,:] = labelMedium[:,:,:params.NUM_CATEGORIES]
            
        batchInd += 1

    imgBatch  = image_batch_preprocess(imgBatch, params, meanVals)

    return imgBatch,labelBatch

def load_batch(inds, trainData, params, mode, meanVals=None):
    """
    Given the batch indices, load the images and ground truth (labels or depth data)
    :param inds: batch indices
    :param trainData: training paths of CLS files (string replacement to get RGB and depth files) and starting x,y pixel positions (can be      non-zero if blocking is set to happen in params.py)
    :param params: input parameters from params.py
    :param mode: either SEMANTIC_MODE or SINGLEVIEW_MODE from params (0 or 1)
    :param meanVals: used for mean subtraction if non-rgb imagery
    :return: numpy arrays for image and ground truth batch data
    """
    
    if params.BLOCK_IMAGES:
        batchShape = (params.BATCH_SZ, params.BLOCK_SZ[0], params.BLOCK_SZ[1])
    else:
        batchShape = (params.BATCH_SZ, params.IMG_SZ[0], params.IMG_SZ[1])
    
    imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], params.NUM_CHANNELS))
    
    numChannels = None
    if mode == params.SINGLEVIEW_MODE:
        numChannels = 1
        labelReplaceStr = params.DEPTH_FILE_STR
    elif mode == params.SEMANTIC_MODE:
        numChannels = params.NUM_CATEGORIES
        labelReplaceStr = params.CLASS_FILE_STR

    labelBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], numChannels))
    
    batchInd = 0
    for i in inds:
        currData = trainData[i]
        imgPath = currData[0]
        imageName = os.path.split(imgPath)[-1]
        labelPath = os.path.join(params.LABEL_DIR,imageName)
        currImg = load_img(imgPath)
        currLabel = get_label_mask_bb(labelPath, params, mode)
        ###make sure the input size is equal to designed size
        if currImg.shape[0]!=batchShape[1] or currImg.shape[1]!=batchShape[2]:
            currImg=cv2.resize(currImg,(batchShape[1],batchShape[2]))
            currLabel=cv2.resize(currLabel,(batchShape[1],batchShape[2]),cv2.INTER_NEAREST)
        
        imageMedium,labelMedium = image_augmentation(currImg, currLabel)

        imgBatch[batchInd,:,:,:] = imageMedium
        if mode == params.SINGLEVIEW_MODE:
            labelBatch[batchInd,:,:,0] = labelMedium
        else:
            labelBatch[batchInd,:,:,:] = labelMedium[:,:,:params.NUM_CATEGORIES]
            
        batchInd += 1

    imgBatch  = image_batch_preprocess(imgBatch, params, meanVals)

    return imgBatch,labelBatch
def get_batch_inds_bb(idx, params):
    """
    Given a list of indices (random sorting happens outside), break into batches of indices for training
    :param idx: list of indices to break
    :param params: input parameters from params.py
    :return: List where each entry contains batch indices to pass through at the current iteration
    """

    N = len(idx)
    batchInds = []
    idx0 = 0
    toProcess = True
    while toProcess:
        idx1 = idx0 + params.BATCH_SZ
        if idx1 > N:
            idx1 = N
            idx0 = idx1 - params.BATCH_SZ
            toProcess = False
        batchInds.append(idx[idx0:idx1])
        idx0 = idx1
    return batchInds


def convert_labels(Lorig, params, toLasStandard=True):
    """
    Convert the labels from the original CLS file to consecutive integer values starting at 0
    :param Lorig: numpy array containing original labels
    :param params: input parameters from params.py
    :param toLasStandard: determines if labels are converted from the las standard labels to training labels
        or from training labels to the las standard
    :return: Numpy array containing the converted labels
    """
    L = Lorig.copy()
    if toLasStandard:
        labelMapping = params.LABEL_MAPPING_TRAIN2LAS
    else:
        labelMapping = params.LABEL_MAPPING_LAS2TRAIN
        
    for key,val in labelMapping.items():
        L[Lorig==key] = val
        
    return L


def get_blocks(params):
    """
    Create blocks using the image dimensions, block size and overlap.
    :param params: input parameters from params.py
    :return: List of start row/col indices of the blocks
    """
    blocks = []
    yEnd,xEnd = np.subtract(params.IMG_SZ, params.BLOCK_SZ)
    x = np.linspace(0, xEnd, np.ceil(xEnd/np.float(params.BLOCK_SZ[1]-params.BLOCK_MIN_OVERLAP))+1, endpoint=True).astype('int')
    y = np.linspace(0, yEnd, np.ceil(yEnd/np.float(params.BLOCK_SZ[0]-params.BLOCK_MIN_OVERLAP))+1, endpoint=True).astype('int')
    
    for currx in x:
        for curry in y:
            blocks.append((currx,curry))
            
    return blocks


def get_train_data(imgPaths, params):
    """
    Create training data containing image paths and block information. If the full image is being used
        then the start row/col values are always 0,0. 
    :param imgPaths: list of image paths to be used for training
    :param params: input parameters from params.py
    :return: List of training data with image paths and block information
    """
    blocks = get_blocks(params)
    trainData = []
    for imgPath in imgPaths:
        for block in blocks:
            trainData.append((imgPath,block[0],block[1]))
            
    return trainData
def load_all_data(trainData,params, mode):
    """
    this is used to load all data into the memory and make training run faster. // i hope so. 
    """
    if params.BLOCK_IMAGES:
            batchShape = (params.BATCH_SZ, params.BLOCK_SZ[0], params.BLOCK_SZ[1])
    else:
        batchShape = (params.BATCH_SZ, params.IMG_SZ[0], params.IMG_SZ[1])

    img_data=[]
    label_data=[]
    for i in range(len(trainData)):
        currData = trainData[i]
        imgPath = currData[0]
        imageName = os.path.split(imgPath)[-1]
        labelPath = os.path.join(params.LABEL_DIR,imageName)
        currImg = load_img(imgPath)
        currLabel = get_label_mask(labelPath, params, mode)
        ###make sure the input size is equal to designed size
        if currImg.shape[0]!=batchShape[1] or currImg.shape[1]!=batchShape[2]:
            currImg=cv2.resize(currImg,(batchShape[1],batchShape[2]))
            currLabel=cv2.resize(currLabel,(batchShape[1],batchShape[2]),cv2.INTER_NEAREST)
        img_data.append(currImg)
        label_data.append(currLabel)
    return img_data, label_data

import tensorflow as tf
from keras import backend as K
def cross_entropy_balanced(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    count_pos = tf.reduce_sum(y_true)
    ratio=count_pos/tf.reduce_sum(count_pos)
    weights=1.0-ratio
    

            # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    
    return loss

def my_weighted_loss(onehot_labels, logits):
    """scale loss based on class weights
    """
    # compute weights based on their frequencies
    beta=tf.reduce_sum(onehot_labels,1);
    beta=tf.reduce_sum(beta,1)
    #beta=tf.count_nonzero(onehot_labels,0)
    num_pixel=K.cast(K.int_shape(logits)[1]*K.int_shape(logits)[2],"float32")
    beta=K.cast(beta, "float32")/num_pixel
#    class_weights=np.empty((onehot_labels.shape[0],onehot_labels.shape[-1]))
#    for i in range(logits.shape[-1]):
#        class_weights[:,i]=1.-beta[:,i];
    class_weights=[[2-2.*beta[:,0]], [2-2.*beta[:,1]]];
    class_weights= tf.transpose(class_weights)
    #class_weights =[1,20]  # set your class weights here
    # computer weights based on onehot labels
    class_weights= tf.expand_dims(class_weights, 1);
 #   class_weights=tf.expand_dims(class_weights, 1);
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=[onehot_labels], logits=[logits])
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss
def my_weighted_loss_5_classes(onehot_labels, logits):
    """scale loss based on class weights
    """
    # compute weights based on their frequencies
    beta=tf.reduce_sum(onehot_labels,1);
    beta=tf.reduce_sum(beta,1)
    #beta=tf.count_nonzero(onehot_labels,0)
    num_pixel=K.cast(K.int_shape(logits)[1]*K.int_shape(logits)[2],"float32")
    beta=K.cast(beta, "float32")/num_pixel
#    class_weights=np.empty((onehot_labels.shape[0],onehot_labels.shape[-1]))
#    for i in range(logits.shape[-1]):
#        class_weights[:,i]=1.-beta[:,i];
    class_weights=[[5-5.*beta[:,0]], [5-5.*beta[:,1]], [5-5.*beta[:,2]], [5-5.*beta[:,3]], [5-5.*beta[:,4]]];
    class_weights= tf.transpose(class_weights)
    #class_weights =[1,20]  # set your class weights here
    # computer weights based on onehot labels
    class_weights= tf.expand_dims(class_weights, 1);
 #   class_weights=tf.expand_dims(class_weights, 1);
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=[onehot_labels], logits=[logits])
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss

def my_class_weighted_loss(onehot_labels, logits):
    """scale loss based on class weights
    """
    num_class=params.NUM_CATEGORIES
    # compute weights based on their frequencies
    beta=tf.reduce_sum(onehot_labels,1);
    beta=tf.reduce_sum(beta,1)
    #beta=tf.count_nonzero(onehot_labels,0)
    num_pixel=K.cast(K.int_shape(logits)[1]*K.int_shape(logits)[2],"float32")
    beta=K.cast(beta, "float32")/num_pixel
    #    class_weights=np.empty((onehot_labels.shape[0],onehot_labels.shape[-1]))
    #    for i in range(logits.shape[-1]):
    #        class_weights[:,i]=1.-beta[:,i];
    class_weights=[]

    for i in range(num_class):
        class_weights.append([num_class-num_class*beta[:,i]])
        #class_weights=[[5-5.*beta[:,0]], [5-5.*beta[:,1]], [5-5.*beta[:,2]], [5-5.*beta[:,3]], [5-5.*beta[:,4]]];
    class_weights= tf.transpose(class_weights)
    #class_weights =[1,20]  # set your class weights here
    # computer weights based on onehot labels
    class_weights= tf.expand_dims(class_weights, 1);
 #   class_weights=tf.expand_dims(class_weights, 1);
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=[onehot_labels], logits=[logits])
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss

def my_weighted_loss_3_classes(onehot_labels, logits):
    """scale loss based on class weights
    """
    # compute weights based on their frequencies
    beta=tf.reduce_sum(onehot_labels,1);
    beta=tf.reduce_sum(beta,1)
    #num_class=K.cast(K.int_shape(logits)[-1])
    total_num=tf.reduce_sum(beta,-1,keep_dims=True)
    total_num=K.cast(total_num,"float32")
    #beta=tf.count_nonzero(onehot_labels,0)
    #total_num=K.cast(K.int_shape(logits)[1]*K.int_shape(logits)[2],"float32")
    beta=K.cast(beta, "float32")/total_num
#    class_weights=np.empty((onehot_labels.shape[0],onehot_labels.shape[-1]))
#    for i in range(logits.shape[-1]):
#        class_weights[:,i]=1.-beta[:,i];
    class_weights=[[3.3-3.3*beta[:,0]], [3.3-3.3*beta[:,1]], [3.3-3.3*beta[:,2]]];
    class_weights= tf.transpose(class_weights)
    #class_weights =[1,20]  # set your class weights here
    # computer weights based on onehot labels
    class_weights= tf.expand_dims(class_weights, 1);
 #   class_weights=tf.expand_dims(class_weights, 1);
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=[onehot_labels], logits=[logits])
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss


def softmax_crossentrop_weighted(y_true, y_pred):
    log_softmax = tf.nn.log_softmax(y_pred)

    # y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    # unpacked = tf.unstack(y_true, axis=-1)
    # y_true = tf.stack(unpacked[:-1], axis=-1)


    count_pos = tf.reduce_sum(y_true)
    ratio=count_pos/tf.reduce_sum(count_pos)
    weights=1.0-ratio

 


    cross_entropy = -K.sum(y_true * log_softmax* weights, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def my_tf_balanced_loss(y_true, y_pred):
    class_weight = {0: 1.,
                1: 5.,
                2: 10.}#,
                #3: 60.,
                #4: 70.}
    return tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred, class_weight, name=None)


if __name__ == '__main__':
    img_folder='G:/programs/dfc2019-master/track1/data/validate/Track1_class_final_merge_cutted'
    glob_path=os.path.join(img_folder,'*.tif')
    files=glob(glob_path)
    from scipy.misc import imread
    for file in files:
        img=imread(file)
        imageMedium,labelMedium=image_augmentation_test(img, img[:,:,0])

        