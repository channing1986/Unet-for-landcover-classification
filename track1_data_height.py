import json
from keras import backend as K
from keras.applications import imagenet_utils
from keras.layers import Dense,Input,merge,Flatten,Dropout,LSTM
from keras.models import Sequential,Model
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from functools import partial
import tensorflow as tf
import os
import cv2
from osgeo import gdal
import params
import tifffile
from dataFunctions import convert_labels,load_img
from DataAugment import normalize_image_to_path,augment_online,convertLas2Train
from scipy.misc import imread as sc_imread
from scipy.misc import imsave as sc_imsave

IMG_FOLDER='img_patch'
LABEL_FOLDER='label_patch'
IMG_TXT='img_list.txt'
LABEL_TXT='label_list.txt'
#DATA_FOLDER='C:/TrainData/Track1/train'
num_workers=6
def input_generator_height(img_files, label_files, class_files,batch_size, extra_files=[],net_name='',num_category=5):
    
    NUM_class=num_category
    channels=net_name.split('_')[1]
    if channels[-1]=='c':
        TRAIN_task='CLS'
    else:
        TRAIN_task='AGL'
    if channels[0:3]=='msi':
        is_MSI=True
        is_msi_data=True
    else:
        is_msi_data=False
    if len(channels)>=4:
        EXTRA_data=True
        if channels[-1]=='c':
            EXTRA_format='c'
        else:
            EXTRA_format='h'


    N = len(img_files) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

   # executor = ProcessPoolExecutor(max_workers=3)

    while True:
        for inds in batchInds:
            img_batch = [img_files[ind] for ind in inds]
            label_batch = [label_files[ind] for ind in inds]
            class_label=[class_files[ind] for ind in inds]
            if len(channels)>3:
                extra_batch=[extra_files[ind] for ind in inds]
            else:
                extra_batch=['']
            imgdata, gts= load_cnn_batch(img_batch, class_label,label_batch,extra_batch,is_msi_data,num_category)#, executor)      
            
            if 0:
                import matplotlib.pyplot as plt 
                plt.subplot(221) #用于显示多个子图121代表行、列、位置
                plt.imshow(imgdata[0,:,:,0:3])
                plt.title('org')
                plt.subplot(222)
                plt.imshow(gts[0,:,:,0])
                plt.title('background') #添加标题
                plt.subplot(223)
                plt.imshow(imgdata[0,:,:,3])
                plt.title('dsm') #添加标题
                plt.subplot(224)
                plt.imshow(imgdata[1,:,:,0:3])
                plt.title('roof') #添加标题
                plt.show()
            yield (imgdata, gts)
            #return (imgdata, gts)

def load_cnn_batch(img_batch, class_label,label_batch, extra_batch, is_msi_data,num_class,executor=''):
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
        currInput['num_class']=num_class
        currInput['class_label']=class_label[i]
        if len(extra_batch[0])>3:
            currInput['extra'] = extra_batch[i]
        else:
            currInput['extra'] = ''
        #futures.append(executor.submit(_load_batch_helper, currInput))
        results.append(_load_batch_helper(currInput))

        

    #results = [future.result() for future in futures]
 
    for  i, result in enumerate(results):
        imgdata.append(result[0][0])
        imgdata.append(result[0][1])
        labels.append(result[1][0])
        labels.append(result[1][1])

    y_train=np.array(labels, np.float32)-10.0
    x_train = np.array(imgdata, np.float32)
    if len(extra_batch[0])>3:
        if not is_msi_data:
            x_train[:,:,:,0:-1]=x_train[:,:,:,0:-1]/125.0-1
        if extra_batch[0][-9:-6]=='AGL':
            x_train[:,:,:,-1]=x_train[:,:,:,-1]/20.0-1
        elif extra_batch[0][-9:-6]=='CLS':
            x_train[:,:,:,-1]=x_train[:,:,:,-1]/2.0-1
    else:
        if not is_msi_data:
            x_train=x_train/125.0-1
    #y_train=np.squeeze(y_train)
    return x_train, y_train
def load_all_data_files_balanced_patches_height(data_folder,vali_ratio=0.1,net_name='unet_rgbc_c'):
    class_folder='label_patch'
    dsm_folder='dsm_patch'
    img_folder='img_patch'
    dsm_folder_p='dsm_patch_p2'
    class_folder_p='class_patch_p2'
    is_extrac_data=False
    extrac_folder=''
    task=''
    extrac_format=''
    if net_name[-1]=='c':
        task='CLS'
    elif net_name[-1]=='h':
        task='AGL'
    else:
        print("wrong net_name")
        return    
    channels=net_name.split('_')[1]
    if len(channels)==4:
        is_extrac_data=True
        if channels[-1]=='c':
            extrac_folder=class_folder_p
            extrac_format='CLS'
        elif channels[-1]=='h':
            extrac_folder=dsm_folder_p
            extrac_format='AGL'
    else:
        is_extrac_data=False
    #balanced_sample_number=1600
    imgs = []
    gts=[]
    extras=[]
    class_label=[]

    imgs_v=[]
    gts_v=[]
    extras_v=[]
    class_label_v=[]
    label_list=os.path.join(data_folder,'label_list.txt')
    img_files=[]
    fp = open(label_list)
    lines = fp.readlines()
    fp.close()
    for line in lines:
        line = line.strip('\n')
        img_files.append(line)

    text_files=[#os.path.join(data_folder,'ground_list.txt'),
                 os.path.join(data_folder,'tree_list.txt'),
                 os.path.join(data_folder,'roof_list.txt'),
                 os.path.join(data_folder,'water_list.txt'),
                os.path.join(data_folder,'bridge_list.txt'),
                ]
    all_ids=[]
    clasee_samples=[]
    for i in range(len(text_files)):
        fp = open(text_files[i])
        lines = fp.readlines()
        fp.close()
        ids=[]
        for line in lines:
            line = line.strip('\n')
            ids.append(int(line))
        all_ids.append(ids)
        clasee_samples.append(len(lines))
    max_samples=max(clasee_samples)
    min_samples=min(clasee_samples)
    record_sampels=0
    #extrac the validation data first
    val_num=200
    val_ids=[]
    for i in range(len(text_files)):
        class_ids=all_ids[i]
        idx = np.random.permutation(len(class_ids))
        ids=idx[0:val_num]
        val_ids.extend(class_ids[ind] for ind in ids)
    train_ids=[]
    val_ids=set(val_ids)
    batch_star=[0,0,0,0,0]
    while record_sampels<=max_samples:
        batch_ids=[]
        
        for i in range(len(text_files)):
            class_ids=all_ids[i]
            batch_end=batch_star[i]+min_samples
            if batch_end>clasee_samples[i]:
                num_plu=batch_end-clasee_samples[i]
                batch_end=clasee_samples[i]
                batch_range1=range(batch_star[i],batch_end)
                batch_ids.extend((class_ids[ind] for ind in batch_range1))
                batch_range2=range(0,num_plu)
                batch_ids.extend((class_ids[ind] for ind in batch_range2))
            else:
                batch_range=np.array(range(batch_star[i],batch_end))
                batch_star[i]=batch_end
                batch_ids.extend(class_ids[ind] for ind in batch_range)
        ##remove the validat ids
        batch_ids=set(batch_ids)
        batch_ids=batch_ids-val_ids
        train_ids.extend(batch_ids)
        record_sampels=record_sampels+min_samples

    for id in train_ids:
        line=img_files[id-1]
        if task=='AGL':
            gts.append(os.path.join(data_folder,dsm_folder,line.replace('CLS',task)))
        elif task=='CLS':
            gts.append(os.path.join(data_folder,class_folder,line))
        img_path=line.replace('CLS','RGB')
        imgs.append(os.path.join(data_folder,img_folder,img_path))
        extra_path=line.replace('CLS',extrac_format)
        extras.append(os.path.join(data_folder,extrac_folder,extra_path))
        class_label_path=os.path.join(data_folder,class_folder,line)
        class_label.append(class_label_path)
    for id in val_ids:
        line=img_files[id-1]
        if task=='AGL':
            gts_v.append(os.path.join(data_folder,dsm_folder,line.replace('CLS',task)))
        elif task=='CLS':
            gts_v.append(os.path.join(data_folder,class_folder,line))
        img_path=line.replace('CLS','RGB')
        imgs_v.append(os.path.join(data_folder,img_folder,img_path))
        extra_path=line.replace('CLS',extrac_format)
        extras_v.append(os.path.join(data_folder,extrac_folder,extra_path))
        class_label_path=os.path.join(data_folder,class_folder,line)
        class_label_v.append(class_label_path)
    if is_extrac_data:
        return imgs,gts,extras,class_label, imgs_v, gts_v, extras_v,class_label_v
    else:
        return imgs, gts,class_label, imgs_v,  gts_v,class_label_v
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
    class_label_file=inputDict['class_label']
    extra_file=inputDict['extra']
    num_class=inputDict['num_class']
    inputs=[]
    labels=[]
    img_data=load_img(rgb_file)
    label_data=load_img(gts_file)
    class_label=load_img(class_label_file)

    if gts_file[-9:-6]=='AGL':
        label_data[np.isnan(label_data)]=params.IGNORE_VALUE
        #label_data[label_data==0]=0.001
    img_data=img_data.astype(np.float)
    if len(extra_file)>3:
        #print('sfd')
        extra_data=load_img(extra_file)
        if extra_file[-9:-6]=='AGL':
            extra_data[np.isnan(extra_data)]=0.001
            #extra_data[extra_data==0]=0.001
        extra_data = extra_data.astype(np.float)



    label_data = label_data.astype(np.float)

    # label_data[label_data==1] = 0
    # label_data[label_data==2] = 1
    # label_data[label_data==3] = 0
    # label_data[label_data==4] = 2
    # label_data[label_data==5] = 3
    ###currLabel=convertLas2Train(label_data, params.LABEL_MAPPING_LAS2TRAIN)
    if 0:
                import matplotlib.pyplot as plt 
                plt.subplot(121) #用于显示多个子图121代表行、列、位置
                plt.imshow(img_data)
                plt.title('org')
                plt.subplot(122)
                plt.imshow(label_data)
                plt.title('rote90') #添加标题
                plt.show()
            
    if gts_file[-9:-6]=='CLS':
        currLabel=np.array(label_data,np.float)
        currLabel = to_categorical(currLabel, num_classes=int(num_class)+1)
        label_data =currLabel[:,:,0:-1]
    else:
        label_data=label_data[:,:,np.newaxis]
        class_label=class_label[:,:,np.newaxis]
        label_data=np.concatenate((label_data, class_label), axis=-1)
    if len(extra_file)>3:
        extra_data = extra_data[:,:,np.newaxis]
        label_data=np.concatenate((label_data, extra_data), axis=-1)
    from dataFunctions import image_augmentation,image_augmentation_test
    imageMedium,labelMedium = image_augmentation_test(img_data, label_data)
    
    if len(extra_file)>3:
        extra_data_m = labelMedium[:,:,-1]
        extra_data_m = extra_data_m[:,:,np.newaxis]
        img_data=np.concatenate((img_data, extra_data), axis=-1)
        imageMedium=np.concatenate((imageMedium, extra_data_m), axis=-1)
        label_data=label_data[:,:,:-1]
        labelMedium=labelMedium[:,:,:-1]
    inputs.append(img_data)
    labels.append(label_data)    
    inputs.append(imageMedium)
    labels.append(labelMedium)
    return inputs, labels
def no_nan_mse_evenloss(y_true, y_pred):
    """
        Custom mean squared error loss function for single-view depth prediction used to ignore NaN/invalid depth values
        :param y_true: ground truth depth
        :param y_pred: predicted depth
        :return: mse loss without nan influence
    """
    y_dsm=y_true[:,:,:,0]
    y_pred=K.squeeze(y_pred,axis=-1)
    y_label=K.cast(y_true[:,:,:,1]+10,'uint8')
    num_class=5

    mask_true = K.cast(K.not_equal(y_dsm, params.IGNORE_VALUE), K.floatx())
    squared_error=K.square((y_dsm-y_pred)*mask_true)
    
    mask_ground=K.cast(K.equal(y_label, 0), K.floatx())#*mask_true
    ground_squared_error=K.sum(mask_ground*squared_error)/K.maximum(K.sum(mask_ground), 1)
    if K.sum(mask_ground)==0:
        num_class=num_class-1
    mask_tree=K.cast(K.equal(y_label, 1), K.floatx())#*mask_true
    tree_squared_error=K.sum(mask_tree*squared_error)/K.maximum(K.sum(mask_tree), 1)
    if K.sum(mask_tree)==0:
        num_class=num_class-1
    mask_roof=K.cast(K.equal(y_label, 2), K.floatx())#*mask_true
    roof_squared_error=K.sum(mask_roof*squared_error)/K.maximum(K.sum(mask_roof), 1)
    if K.sum(mask_roof)==0:
        num_class=num_class-1
    mask_water=K.cast(K.equal(y_label, 3), K.floatx())#*mask_true
    water_squared_error=K.sum(mask_water*squared_error)/K.maximum(K.sum(mask_water), 1)
    if K.sum(mask_water)==0:
        num_class=num_class-1

    mask_bridge=K.cast(K.equal(y_label, 4), K.floatx())#*mask_true
    bridge_squared_error=K.sum(mask_bridge*squared_error)/K.maximum(K.sum(mask_bridge), 1)
    if K.sum(mask_bridge)==0:
        num_class=num_class-1
    masked_mse = (ground_squared_error+1.5*tree_squared_error+1.5*roof_squared_error+water_squared_error+1.5*bridge_squared_error)
    masked_mse=masked_mse/num_class
    return masked_mse

def no_nan_mse_evenloss_bridge(y_true, y_pred):
    """
        Custom mean squared error loss function for single-view depth prediction used to ignore NaN/invalid depth values
        :param y_true: ground truth depth
        :param y_pred: predicted depth
        :return: mse loss without nan influence
    """
    y_dsm=y_true[:,:,:,0]
    y_pred=K.squeeze(y_pred,axis=-1)
    y_label=K.cast(y_true[:,:,:,1]+10,'uint8')

    mask_true = K.cast(K.not_equal(y_dsm, params.IGNORE_VALUE), K.floatx())
    squared_error=K.square((y_dsm-y_pred)*mask_true)
    num_class=2
    mask_ground=K.cast(K.equal(y_label, 0), K.floatx())#*mask_true
    ground_squared_error=K.sum(mask_ground*squared_error)/K.maximum(K.sum(mask_ground), 1)
    if K.sum(mask_ground)==0:
        num_class=num_class-1

    mask_bridge=K.cast(K.equal(y_label, 4), K.floatx())#*mask_true
    bridge_squared_error=K.sum(mask_bridge*squared_error)/K.maximum(K.sum(mask_bridge), 1)
    if K.sum(mask_bridge)==0:
        num_class=num_class-1
    masked_mse = (ground_squared_error+2*bridge_squared_error)
    masked_mse=masked_mse/np.maximum(1,num_class)
    return masked_mse
def no_nan_mse(y_true, y_pred):
    """
        Custom mean squared error loss function for single-view depth prediction used to ignore NaN/invalid depth values
        :param y_true: ground truth depth
        :param y_pred: predicted depth
        :return: mse loss without nan influence
    """
    y_dsm=y_true[:,:,:,0]
    y_label=K.cast(y_true[:,:,:,1]+10,'uint8')
    onehot_labels = K.one_hot(y_label, 5)
    beta=tf.reduce_sum(onehot_labels,1);
    beta=tf.reduce_sum(beta,1)
    num_pixel=K.cast(K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2],"float32")
    #beta=tf.count_nonzero(onehot_labels,0)
    beta=K.cast(beta, "float32")/num_pixel

    class_weights=[[5-5.*beta[:,0]], [5-5.*beta[:,1]], [5-5.*beta[:,2]], [5-5.*beta[:,3]], [5-5.*beta[:,4]]];
    class_weights= tf.transpose(class_weights)

    class_weights= tf.expand_dims(class_weights, 1);
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
    weights=tf.expand_dims(weights,-1)

    y_dsm= tf.expand_dims(y_dsm, -1)
    mask_true = K.cast(K.not_equal(y_dsm, params.IGNORE_VALUE), K.floatx())
    masked_squared_error = K.square(mask_true * (y_dsm - y_pred))
    masked_squared_error=weights*masked_squared_error
    masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
    return masked_mse

def input_generator_RGB(img_files, dsm_files,label_files, batch_size):
    """
    """

    N = len(img_files) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    executor = ProcessPoolExecutor(max_workers=num_workers)

    while True:
        for inds in batchInds:
            img_batch = [img_files[ind] for ind in inds]
            label_batch = [label_files[ind] for ind in inds]
            dsm_batch = [dsm_files[ind] for ind in inds]
            imgdata, gts= load_cnn_batch_RGB(img_batch, dsm_batch,label_batch,  executor)      
            yield (imgdata, gts)
            #return (imgdata, gts)

def load_cnn_batch_RGB(img_batch, dsm_batch, label_batch,  executor):
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
        currInput['dsm'] = dsm_batch[i] 
        futures.append(executor.submit(_load_batch_helper_RGB, currInput))
 #       result=_load_batch_helper_RGB(currInput)

        

    results = [future.result() for future in futures]
 
    for  i, result in enumerate(results):
        imgdata.append(result[0][0])
        imgdata.append(result[0][1])
        labels.append(result[1][0])
        labels.append(result[1][1])

    #imgdata = imagenet_utils.preprocess_input(imgdata)
    #imgdata = imgdata / 255.0


    normalize_method=3
    if (normalize_method==1):
        y_train = np.array(labels, np.float32)/125.0-1.0
    #y_train=labels
        x_train = np.array(imgdata, np.float32)/125.0-1.0
    elif (normalize_method==2):
        y_train=np.array(labels, np.float32)
        x_train = np.array(imgdata, np.float32)
    elif(normalize_method==3):
        y_train=np.array(labels, np.float32)
        
        x_train = np.array(imgdata, np.float32)
        x_train=x_train/125.0-1


    return x_train, y_train


def _load_batch_helper_RGB(inputDict):
    """
    Helper for load_cnn_batch that actually loads imagery and supports parallel processing
    :param inputDict: dict containing the data and metadataStats that will be used to load imagery
    :return currOutput: dict with image data, metadata, and the associated label
    """
    #print("fsf")
    #return
    rgb_file = inputDict['rgb']
    gts_file = inputDict['gts']
    dsm_file=inputDict['dsm']

    inputs=[]
    labels=[]
    img_data=load_img(rgb_file)
    label_data=load_img(gts_file)
    dsm_data=load_img(dsm_file)
    dsm_data[np.isnan(dsm_data)] = params.IGNORE_VALUE

    img_data=img_data.astype(np.float)
    label_data = label_data.astype(np.float)
    dsm_data=dsm_data[:,:,np.newaxis]
    label_data=label_data[:,:,np.newaxis]
    rgbh=np.concatenate((dsm_data, label_data), axis=-1)
    #currLabel=convertLas2Train(label_data, params.LABEL_MAPPING_LAS2TRAIN)
    #currLabel=np.array(label_data,np.float)

    #aa=currLabel==2
 #   currLabel = to_categorical(currLabel, num_classes=5+1)
 #   currLabel =currLabel[:,:,0:-1]
    from dataFunctions import image_augmentation
    imageMedium,dsmMedium = image_augmentation(img_data, rgbh)
    inputs.append(img_data)
    labels.append(rgbh)    
    inputs.append(imageMedium)
    labels.append(dsmMedium)
    return inputs, labels




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
def load_all_data_files_train_single_class(text_files_positive,text_files_negative,Dataset_path):
    balanced_sample_number=1600
    imgs = []
    gts=[]
    dsm=[]
    imgs_v=[]
    gts_v=[]
    dsm_v=[]
    text_files=[text_files_positive,
                text_files_negative
                ]
    for i in range(len(text_files)):
        fp = open(text_files[i])
        lines = fp.readlines()
        fp.close()
        if i==1:
            balanced_sample_number=round(balanced_sample_number*0.3)
        num_sample=len(lines)
        if num_sample>balanced_sample_number:
        #if 1 :    
            idx = np.random.permutation(num_sample-1) #shuffle the order
            ids=idx[0:balanced_sample_number]
            files = [lines[ind] for ind in ids]
        else:
            idx = np.random.permutation(num_sample-1)
            ids=idx
            files = [lines[ind] for ind in ids]
        count=0
        val_range=int(0.9*len(files))
        for line in files:
            line = line.strip('\n')
            if count<val_range:

                gts.append(os.path.join(Dataset_path,label_folder,line))
                img_path=line.replace('CLS','ortho')
                imgs.append(os.path.join(Dataset_path,img_folder,img_path))
                dsm_path=line.replace('CLS','DSM')
                dsm.append(os.path.join(Dataset_path,dsm_folder,dsm_path))
            else:
                gts_v.append(os.path.join(Dataset_path,label_folder,line))
                img_path=line.replace('CLS','ortho')
                imgs_v.append(os.path.join(Dataset_path,img_folder,img_path))
                dsm_path=line.replace('CLS','DSM')
                dsm_v.append(os.path.join(Dataset_path,dsm_folder,dsm_path)) 
            count=count+1               
    
    #return imgs[val_range:len(imgs)], gts[val_range:len(imgs)], imgs[0:val_range], gts[0:val_range]
    return imgs, dsm,gts, imgs_v, dsm_v, gts_v
def load_all_data_files_balanced_patches(data_folder,vali_ratio=0.1):
    dsm_folder='dsm_patch'
    img_folder='img_patch'
    label_folder='label_patch'
    balanced_sample_number=1600
    imgs = []
    label=[]
    dsm=[]

    imgs_v=[]
    label_v=[]
    dsm_v=[]
    text_files=[os.path.join(data_folder,'ground_list.txt'),
                os.path.join(data_folder,'tree_list.txt'),
                os.path.join(data_folder,'roof_list.txt'),
                os.path.join(data_folder,'water_list.txt'),
                os.path.join(data_folder,'bridge_list.txt'),
                ]
    for i in range(len(text_files)):
        fp = open(text_files[i])
        lines = fp.readlines()
        fp.close()
        num_sample=len(lines)
        if num_sample>balanced_sample_number:
            idx = np.random.permutation(num_sample-1) #shuffle the order
            ids=idx[0:balanced_sample_number]
            files = [lines[ind] for ind in ids]
        else:
            idx = np.random.permutation(num_sample-1)
            ids=idx
            files = [lines[ind] for ind in ids]
        count=0
        val_range=int(0.9*len(files))
        for line in files:
            line = line.strip('\n')
            if count<val_range:
                label.append(os.path.join(data_folder,label_folder,line))
                dsm_path=line.replace('CLS','AGL')
                dsm.append(os.path.join(data_folder,dsm_folder,dsm_path))
                img_path=line.replace('CLS','RGB')
                imgs.append(os.path.join(data_folder,img_folder,img_path))
            else:
                label_v.append(os.path.join(data_folder,label_folder,line))
                dsm_path=line.replace('CLS','AGL')
                dsm_v.append(os.path.join(data_folder,dsm_folder,dsm_path))
                img_path=line.replace('CLS','RGB')
                imgs_v.append(os.path.join(data_folder,img_folder,img_path))
            count=count+1               
    
    #return imgs[val_range:len(imgs)], gts[val_range:len(imgs)], imgs[0:val_range], gts[0:val_range]
    return imgs, dsm,label,imgs_v, dsm_v, label_v
def load_all_data_files(data_folder,vali_ratio=0.1):

    ortho_txt=os.path.join(data_folder,IMG_TXT)
    label_txt=os.path.join(data_folder,LABEL_TXT)
    orthos=[]
    label=[]
    fp = open(ortho_txt)
    lines = fp.readlines()
    for line in lines:
            line = line.strip('\n')           
            orthos.append(os.path.join(data_folder,IMG_FOLDER,line))
    fp.close()
    fp = open(label_txt)
    lines = fp.readlines()
    for line in lines:
            line = line.strip('\n')           
            label.append(os.path.join(data_folder,LABEL_FOLDER,line))
    fp.close()
    if len(orthos)!=len(label):
        return [], [],[],[]
    num=round(len(orthos)*(1-vali_ratio))
    return orthos[:num],label[:num],orthos[num:],label[num:]

def input_generator_online_process(img_data,label_data,batch_size,path_size,overlap):
    """
    """

    N = len(img_data) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    executor = ProcessPoolExecutor(max_workers=6)

    while True:
        for inds in batchInds:
            img_batch = [img_data[ind] for ind in inds]
            label_batch = [label_data[ind] for ind in inds]
            imgdata, gts= load_cnn_batch_online(img_batch, label_batch, executor)      
           # yield (imgdata, gts)

            return (imgdata, gts)

def load_cnn_batch_online(img_batch, label_batch,  executor):
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

        futures.append(executor.submit(_load_batch_helper_online, currInput))
 #       result=_load_batch_helper_online(currInput)

        

    results = [future.result() for future in futures]
       # results.append( _load_batch_helper(currInput))
    for  i, result in enumerate(results):
        imgdata.extend(result[0])
 #       imgdata.append(result[0][1])
        labels.extend(result[1])
 #       labels.append(result[1][1])




    # for i in range (len(imgdata)):
    #     img_write_path='img_'+str(i)+'.tif'
    #     label_write_path='label_'+str(i)+'.tif'
    #     cv2.imwrite(img_write_path,imgdata[i])
    #         #sio.savemat(label_write_path,mdict={'label_map':labels[i]})
    #     cv2.imwrite(label_write_path,labels[i])


    normalize_method=3
    if (normalize_method==1):
        y_train = np.array(labels, np.float32)/125.0-1.0
    #y_train=labels
        x_train = np.array(imgdata, np.float32)/125.0-1.0
    elif (normalize_method==2):
        y_train=np.array(labels, np.float32)
        x_train = np.array(imgdata, np.float32)
    elif(normalize_method==3):
        y_train=np.array(labels, np.float32)
        x_train = np.array(imgdata, np.float32)
        x_train=x_train/256.0-1
 

    return x_train, y_train


def _load_batch_helper_online(inputDict):
    """
    Helper for load_cnn_batch that actually loads imagery and supports parallel processing
    :param inputDict: dict containing the data and metadataStats that will be used to load imagery
    :return currOutput: dict with image data, metadata, and the associated label
    """
    #print("fsf")
    #return
    rgb_file = inputDict['rgb']
    gts_file = inputDict['gts']


    inputs=[]
    labels=[]

    from dataFunctions import image_augmentation
    imageMedium,labelMedium = data_patch_augment_online(rgb_file, gts_file)
    # inputs.append(rgb_file)
    # inputs.append(imageMedium)
    # labels.append(gts_file)
    # labels.append(labelMedium)

    return imageMedium, labelMedium



def input_load_train_data(img_train,lable_train):
    inputs=[]
    labels=[]
    from dataFunctions import load_img
    from DataAugment import convertLas2Train
    for i in range(len(img_train)):
        rgb_file=img_train[i]
        gts_file=lable_train[i]
        imgPath = os.path.join(DATA_FOLDER,IMG_FOLDER,rgb_file)
        labelPath = os.path.join(DATA_FOLDER,LABEL_FOLDER,gts_file)
        currImg = load_img(imgPath)
#        currImg=currImg/256.0-1
        label=load_img(labelPath)
        label=convertLas2Train(label, params.LABEL_MAPPING_LAS2TRAIN)
        inputs.append(currImg)
        labels.append(label)
    return inputs,labels

def find_all_test_data(data_folder):
    imgs=[]
    labels=[]
    img_folder=os.path.join(data_folder,'Track1-RGB')
    img_files = glob.glob(os.path.join(img_folder, '*.tif'))
    for imgPath in img_files:
        imageName = os.path.split(imgPath)[-1]
        label=imageName[0:-7]+'CLS.tif'
        imgs.append(imageName)
        labels.append(label)
    ###########################
    ortho_list_file=os.path.join(data_folder,'train_img.txt')
    dsm_list_file=os.path.join(data_folder,'train_label.txt')

    f_ortho = open(ortho_list_file,'w')
    f_dsm = open(dsm_list_file,'w')

    for i in range(len(imgs)):
        f_ortho.write(imgs[i]+'\n');
        f_dsm.write(labels[i]+'\n');
    f_ortho.close()
    f_dsm.close()    

def GetPredictData(ortho_path, path_size=(256,256),overlap_ratio=0.5,img_ioa=[]):
    img_aug=[]
    from DataAugment import normalize_image_to_path,getAuger_p,getAuger_p_p
    #path_size=(256,256)
   # overlap_ratio=0.5
    dsm_path=ortho_path
    [imgs,dsms,image_size]=normalize_image_to_path(ortho_path,dsm_path,path_size,overlap_ratio,work_region=img_ioa,convertLab=False,pad_edge=1,normalize_dsm=1, image_order=1)
    contrast_range_2=(0.9,1.1)
    for i in range(len(imgs)):
        img_data=imgs[i]
        img_data = img_data.transpose(1,2,0)
        img_data=img_data.astype(np.float32)
        for a_id in range (4):
            
            auger=getAuger_p_p(a_id)
            img_a=auger.augment_image(img_data)
            img_aug.append(img_a)
        # for a_id in range (4):
        #     auger=getAuger_p_p(a_id)
        #     img_a=auger.augment_image(rgbh)
        #     img_aug.append(img_a)
    return img_aug,image_size
def load_all_data_test(data_folder,img_folder_name='imgs'):
    imgs=[]
    pathes=[]
    img_folder=os.path.join(data_folder,img_folder_name)
    img_files = glob.glob(os.path.join(img_folder, '*RGB.tif'))
    for imgPath in img_files:
        imageName = os.path.split(imgPath)[-1]
        imgs.append(imageName)
        pathes.append(imgPath)
    ###########################
    ortho_list_file=os.path.join(data_folder,'test_img_list.txt')

    f_ortho = open(ortho_list_file,'w')
 
    for i in range(len(imgs)):
        f_ortho.write(imgs[i]+'\n');
    f_ortho.close()
    return pathes
def data_patch_augment_online(img,label):
    img_patches=[]
    label_patches=[]
    path_size=(512,512)
    overlap_ratio=0.0
    [imgs,labels]=normalize_image_to_path(img,label,path_size,overlap_ratio,convertLab=False,pad_edge=1,normalize_dsm=0)
    for idx in range(len(imgs)):
        au_imgs,au_labels=augment_online(imgs[idx],labels[idx])
        img_patches.extend(au_imgs)
        label_patches.extend(au_labels)

    return img_patches,label_patches;
def Merge_temparal_results(result_folder,out_folder,offset_folder,class_id=-1):
    site_images=[]
    site_names=[]
    if os.path.exists(out_folder)==0:
        os.makedirs(out_folder)
    else:
        glob_path=os.path.join(out_folder,'*.tif')
        files=glob.glob(glob_path) 
        for file in files:
            os.remove(file)
    glob_path=os.path.join(result_folder,'*.tif')
    files=glob.glob(glob_path)
    for img in files:
        image_name=os.path.split(img)[-1]
        site_name=image_name[0:7]
        new_site=True
        for i in range(len(site_names)):
            if site_name==site_names[i]:
                new_site=False
                site_images[i].append(img)
        if new_site:
            site_names.append(site_name)
            site_images.append([img])
 #              site_images[len(site_names)-1].append(img)
    if class_id>=0:
        NUM_CATEGORIES=2
    else:
        NUM_CATEGORIES=params.NUM_CATEGORIES
    for m in range(len(site_names)):
        imgs=site_images[m]
        im=cv2.imread(imgs[0],0)
        vote_map=np.zeros((im.shape[0],im.shape[1],NUM_CATEGORIES))
        for img_p in imgs:
            im=cv2.imread(img_p,0)
            one_hot=to_categorical(im,NUM_CATEGORIES)
            for i in range(vote_map.shape[-1]):
                vote_map[:,:,i]=vote_map[:,:,i]+one_hot[:,:,i]
        pred=np.argmax(vote_map,axis=-1).astype('uint8')
        if pred.shape[0]>512 or pred.shape[1]>512:
            offset_file=os.path.join(offset_folder,site_names[m]+'_DSM.txt')
            offset = np.loadtxt(offset_file)
            offset=offset.astype('int')
            pred=pred[offset[1]:offset[1]+512,offset[0]:offset[0]+512]
        if class_id<0:
            pred=convert_labels(pred,params,toLasStandard=True)
        else:
            pred[pred==1]=class_id
        out_path=os.path.join(out_folder,site_names[m]+'_CLS.tif')
        tifffile.imsave(out_path,pred,compress=6)

def Merge_all_results_with_baseline(result_folders,out_folder=''):
    if len(out_folder)<2:
        out_folder=result_folders[0]
    if os.path.exists(out_folder)==0:
        os.mkdir(out_folder)
    
    result_my=result_folders[0]
    glob_path=os.path.join(result_my,'*RGB.tif')
    files=glob.glob(glob_path)
    baseline_folder=result_folders[1]
   # bridge_folder=result_folders[2]
    for img in files:
        image_name=os.path.split(img)[-1]
        site_name=image_name[0:-7]
        result_my=cv2.imread(img,0)
        #result_my=convert_labels(result_my,params,toLasStandard=False)
        base_file_name=site_name+'CLS.tif'
        result_baseline=cv2.imread(os.path.join(baseline_folder,base_file_name),0)
        result_baseline=convert_labels(result_baseline,params,toLasStandard=False)

#        result_bridge=cv2.imread(os.path.join(bridge_folder,image_name),0)
 #       result_bridge=convert_labels(result_bridge,params,toLasStandard=False)

        result_merge=np.zeros([result_baseline.shape[0],result_baseline.shape[1]], dtype = 'uint8')
 #           result_merge[(result_baseline==0)]=result_baseline[(result_baseline==0)]
        result_merge[(result_baseline==2)]=result_baseline[(result_baseline==2)]
        result_merge[(result_my==2)]=result_my[(result_my==2)]
        #result_merge[(result_baseline==1)]=result_baseline[(result_baseline==1)]
        result_merge[(result_my==1)]=result_my[(result_my==1)]
        #result_merge[(result_baseline==3)]=result_baseline[(result_baseline==3)]
        result_merge[(result_my==3)]=result_my[(result_my==3)]
       # result_merge[(result_baseline==4)]=result_baseline[(result_baseline==4)]
        result_merge[(result_my==4)]=result_my[(result_my==4)]
        #result_merge[(result_bridge==4)]=result_bridge[(result_bridge==4)]
        result_merge=convert_labels(result_merge,params,toLasStandard=True)
        out_path=os.path.join(out_folder,site_name+'CLS.tif')
        tifffile.imsave(out_path,result_merge,compress=6)
        aa=sc_imread(out_path)
        sc_imsave(out_path,aa)

def get_nb_patch(img_dim, patch_size, image_data_format):
    
    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"

    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
#        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])
        img_dim_disc = (img_dim[0]+1, patch_size[0], patch_size[1])

    elif image_data_format == "channels_last":
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
  #      img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1]+1)

    return nb_patch, img_dim_disc
            
if __name__ == '__main__':
    # data_folder='C:/TrainData/Track1/train'
    # find_all_test_data(data_folder)
    pre=[]
    truth=[]
    sess = tf.Session()
    print(sess.run(no_nan_mse_evenloss()))
 