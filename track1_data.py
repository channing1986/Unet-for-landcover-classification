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
 
def input_generator(img_files, label_files, batch_size, dsm_files=[],channels=''):

    N = len(img_files) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    executor = ProcessPoolExecutor(max_workers=num_workers)

    while True:
        for inds in batchInds:
            img_batch = [img_files[ind] for ind in inds]
            label_batch = [label_files[ind] for ind in inds]
            if len(channels)>3:
                dsm_batch=[dsm_files[ind] for ind in inds]
            else:
                dsm_batch=[]
            imgdata, gts= load_cnn_batch(img_batch, label_batch,dsm_batch, channels,executor)      
            yield (imgdata, gts)
            #return (imgdata, gts)

def load_cnn_batch(img_batch, label_batch, dsm_batch, channels,executor):
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

        futures.append(executor.submit(_load_batch_helper, currInput))
 #       result=_load_batch_helper_RGB(currInput)

        

    results = [future.result() for future in futures]
 
    for  i, result in enumerate(results):
        imgdata.append(result[0][0])
        imgdata.append(result[0][1])
        labels.append(result[1][0])
        labels.append(result[1][1])

    y_train=np.array(labels, np.float32)
    x_train = np.array(imgdata, np.float32)
    if channels>3:
        
        x_train[:,:,:,0:-1]=x_train[:,:,:,0:-1]/125.0-1
        if channels[-1]=='h':
            x_train[:,:,:,-1]=x_train[:,:,:,-1]/20.0-1
        elif channels[-1]=='c':
            x_train[:,:,:,-1]=x_train[:,:,:,-1]/params.NUM_CATEGORIES-1
    else:
        x_train=x_train/125.0-1


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
    dsm_file=inputDict['dsm']
    RGBH=1
    if len(dsm_file)<4:
        RGBH=0
    inputs=[]
    labels=[]
    img_data=load_img(rgb_file)
    label_data=load_img(gts_file)
    img_data=img_data.astype(np.float)
    if RGBH:
        dsm_data=load_img(dsm_file)
        dsm_data = dsm_data.astype(np.float)
        dsm_data = dsm_data[:,:,np.newaxis]
        img_data=np.concatenate((img_data, dsm_data), axis=-1)


    label_data = label_data.astype(np.float)
    #currLabel=convertLas2Train(label_data, params.LABEL_MAPPING_LAS2TRAIN)
    currLabel=np.array(label_data,np.float)
    #aa=currLabel==2
    currLabel = to_categorical(currLabel, num_classes=5+1)
    currLabel =currLabel[:,:,0:-1]
    from dataFunctions import image_augmentation
    imageMedium,labelMedium = image_augmentation(img_data, currLabel)
    inputs.append(img_data)
    labels.append(currLabel)    
    inputs.append(imageMedium)
    labels.append(labelMedium)
    return inputs, labels

def input_generator_RGB(img_files, label_files, batch_size):
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
            imgdata, gts= load_cnn_batch_RGB(img_batch, label_batch,  executor)      
            yield (imgdata, gts)
            #return (imgdata, gts)

def load_cnn_batch_RGB(img_batch, label_batch,  executor):
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


    inputs=[]
    labels=[]
    img_data=load_img(rgb_file)
    label_data=load_img(gts_file)

    
    img_data=img_data.astype(np.float)
    label_data = label_data.astype(np.float)
    #currLabel=convertLas2Train(label_data, params.LABEL_MAPPING_LAS2TRAIN)
    currLabel=np.array(label_data,np.float)
    #aa=currLabel==2
    currLabel = to_categorical(currLabel, num_classes=5+1)
    currLabel =currLabel[:,:,0:-1]
    from dataFunctions import image_augmentation
    imageMedium,labelMedium = image_augmentation(img_data, currLabel)
    inputs.append(img_data)
    labels.append(currLabel)    
    inputs.append(imageMedium)
    labels.append(labelMedium)
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
def load_all_data_files_balanced_patches_oldd(data_folder,vali_ratio=0.1):
    label_folder='label_patch'
    img_folder='img_patch'
    balanced_sample_number=1600
    imgs = []
    gts=[]

    imgs_v=[]
    gts_v=[]

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

                gts.append(os.path.join(data_folder,label_folder,line))
                img_path=line.replace('CLS','RGB')
                imgs.append(os.path.join(data_folder,img_folder,img_path))
            else:
                gts_v.append(os.path.join(data_folder,label_folder,line))
                img_path=line.replace('CLS','RGB')
                imgs_v.append(os.path.join(data_folder,img_folder,img_path))
            count=count+1               
    
    #return imgs[val_range:len(imgs)], gts[val_range:len(imgs)], imgs[0:val_range], gts[0:val_range]
    return imgs, gts, imgs_v,  gts_v

def load_all_data_files_balanced_patches(data_folder,vali_ratio=0.1,net_name='unet_rgbc_c'):
    class_folder='label_patch'
    dsm_folder='dsm_patch'
    img_folder='img_patch'
    dsm_folder_p='dsm_patch_p'
    class_folder_p='class_patch_p'
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
        print("wrong net_name")
        return
    #balanced_sample_number=1600
    imgs = []
    gts=[]
    extras=[]

    imgs_v=[]
    gts_v=[]
    extras_v=[]

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
                if task=='AGL':
                    gts.append(os.path.join(data_folder,dsm_folder,line.replace('CLS',task)))
                elif task=='CLS':
                    gts.append(os.path.join(data_folder,class_folder,line))
                img_path=line.replace('CLS','RGB')
                imgs.append(os.path.join(data_folder,img_folder,img_path))
                extra_path=line.replace('CLS',extrac_format)
                extras.append(os.path.join(data_folder,extrac_folder,extra_path))
            else:
                if task=='AGL':
                    gts_v.append(os.path.join(data_folder,dsm_folder,line.replace('CLS',task)))
                elif task=='CLS':
                    gts_v.append(os.path.join(data_folder,class_folder,line))
                img_path=line.replace('CLS','RGB')
                imgs_v.append(os.path.join(data_folder,img_folder,img_path))
                extra_path=line.replace('CLS',extrac_format)
                extras_v.append(os.path.join(data_folder,extrac_folder,extra_path))
            count=count+1               
    if is_extrac_data:
        return imgs,gts,extras,  imgs_v, gts_v, extras_v
    else:
        return imgs, gts, imgs_v,  gts_v
def load_all_data_files_balanced_patches_old(data_folder,vali_ratio=0.1,net_name='unet_rgbc_c'):
    class_folder='label_patch'
    dsm_folder='dsm_patch'
    img_folder='img_patch'
    dsm_folder_p='dsm_patch_p'
    class_folder_p='class_patch_p'
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
        print("wrong net_name")
        return
    balanced_sample_number=1600
    imgs = []
    gts=[]
    extras=[]

    imgs_v=[]
    gts_v=[]
    extras_v=[]

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
                if task=='AGL':
                    gts.append(os.path.join(data_folder,dsm_folder,line.replace('CLS',task)))
                elif task=='CLS':
                    gts.append(os.path.join(data_folder,class_folder,line))
                img_path=line.replace('CLS','RGB')
                imgs.append(os.path.join(data_folder,img_folder,img_path))
                extra_path=line.replace('CLS',extrac_format)
                extras.append(os.path.join(data_folder,extrac_folder,extra_path))
            else:
                if task=='AGL':
                    gts_v.append(os.path.join(data_folder,dsm_folder,line.replace('CLS',task)))
                elif task=='CLS':
                    gts_v.append(os.path.join(data_folder,class_folder,line))
                img_path=line.replace('CLS','RGB')
                imgs_v.append(os.path.join(data_folder,img_folder,img_path))
                extra_path=line.replace('CLS',extrac_format)
                extras_v.append(os.path.join(data_folder,extrac_folder,extra_path))
            count=count+1               
    if is_extrac_data:
        return imgs,gts,extras,  imgs_v, gts_v, extras_v
    else:
        return imgs, gts, imgs_v,  gts_v


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

def input_generator_online_process(img_data,label_data,batch_size,num_class):
    """
    """

    N = len(img_data) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    while True:
        # imgdata=[]
        # gts=[]
        for inds in batchInds:
            img_batch = [img_data[ind] for ind in inds]
            label_batch = [label_data[ind] for ind in inds]
            label_batch=to_categorical(label_batch,num_class+1)
            label_batch=label_batch[:,:,:,:-1]
            label_batch=np.array(label_batch)
            img_batch=np.array(img_batch)

            #imgdata, gts= load_cnn_batch_online(img_batch, label_batch, executor)      
            yield (img_batch, label_batch)

            #return (img_batch, label_batch)
def input_generator_online_process_OLD(img_data,label_data,batch_size,path_size,overlap):
    N = len(img_data) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    while True:
        imgdata=[]
        gts=[]
        for inds in batchInds:
            img_batch = [img_data[ind] for ind in inds]
            label_batch = [label_data[ind] for ind in inds]
            #label_batch=to_categorical(label_batch,num_class+1)
            #label_batch=label_batch[:,:,:0:-1]
            #imgdata, gts= load_cnn_batch_online(img_batch, label_batch, executor)      
           # yield (imgdata, gts)

            return (img_batch, gts)
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
        # imgPath = os.path.join(DATA_FOLDER,IMG_FOLDER,rgb_file)
        # labelPath = os.path.join(DATA_FOLDER,LABEL_FOLDER,gts_file)
        currImg = load_img(rgb_file)
        currImg=currImg/125.0-1
        label_data=load_img(gts_file)
        label_data[label_data==1] = 0
        label_data[label_data==2] = 1
        label_data[label_data==3] = 0
        label_data[label_data==4] = 2
        label_data[label_data==5] = 3
        #label=convertLas2Train(label, params.LABEL_MAPPING_LAS2TRAIN)
        inputs.append(currImg)
        labels.append(label_data)
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


def load_all_data_test_val(data_folder,channels='rgb',track='track1'):
    imgs=[]
    pathes=[]
    site_images=[]
    site_names=[]

    img_folder=os.path.join(data_folder,'imgs')
    if channels[:3]=='rgb':        
        img_files = glob.glob(os.path.join(img_folder, '*RGB.tif'))
    elif channels[:3]=='msi':
        img_files = glob.glob(os.path.join(img_folder, '*MSI.tif'))

    for imgPath in img_files:
        imageName = os.path.split(imgPath)[-1]
        if track=='track3' or track=='track1':
            site_name=imageName[0:7]
        elif track=='track2':
            site_name=imageName[0:11]
        new_site=True
        for i in range(len(site_names)):
            if site_name==site_names[i]:
                new_site=False
                site_images[i].append(imgPath)
        if new_site:
            site_names.append(site_name)
            site_images.append([imgPath])
#              site_images[len(site_names)-1].append(img)
    NUM_CATEGORIES=params.NUM_CATEGORIES
    for m in range(len(site_names)):
        imgs=site_images[m]
        num_imgs=len(imgs)
        idx = np.random.permutation(num_imgs)
        pathes.append(imgs[idx[0]])
        #pathes.append(imgs[idx[1]])

        #pathes.append(imageName)
        
    ###########################
    return pathes    


def load_all_data_test(data_folder,channels='rgb'):
    imgs=[]
    pathes=[]
    #img_folder=os.path.join(data_folder,'epipolar image')
    img_folder=os.path.join(data_folder,'imgs')
    if channels[:3]=='rgb':        
        img_files = glob.glob(os.path.join(img_folder, '*RGB.tif'))
    elif channels[:3]=='msi':
        img_files = glob.glob(os.path.join(img_folder, '*MSI.tif'))

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
    
    result_base=result_folders[0]
    glob_path=os.path.join(result_base,'*CLS.tif')
    files=glob.glob(glob_path)
    rgbh_folder=result_folders[1]
    last_folder=result_folders[2]
    if len(result_folders)>=4:
        use_msi_tree=True
        msitree_folder=result_folders[2]
    else:
        use_msi_tree=False

   # bridge_folder=result_folders[2]
    for img in files:
        image_name=os.path.split(img)[-1]
        site_name=image_name[0:-7]
        result_base=cv2.imread(img,0)
        ###
        # result_base[result_base==50]=1
        # result_base[result_base==100]=2
        # result_base[result_base==150]=3
        # result_base[result_base==200]=4
        ###

        #result_base=cv2.resize(result_base,(1024,1024),cv2.INTER_NEAREST)
        result_base=convert_labels(result_base,params,toLasStandard=False)
        #image_name=site_name+'MSI.tif'
        result_rgbh=cv2.imread(os.path.join(rgbh_folder,image_name),0)
        result_rgbh=convert_labels(result_rgbh,params,toLasStandard=False)

        result_last=cv2.imread(os.path.join(last_folder,image_name),0)
        result_last=convert_labels(result_last,params,toLasStandard=False)
        ###
        #result_rgbh[result_rgbh==1]=1
        # result_base[result_base==100]=2
        # result_base[result_base==150]=3
        # result_base[result_base==200]=4
        ###

        ###ground
        result_merge=np.zeros([result_base.shape[0],result_base.shape[1]], dtype = 'uint8')
 #           result_merge[(result_baseline==0)]=result_baseline[(result_baseline==0)]

        ### roof
        #lds=np.where((result_base==2)&(result_last==2))
        #result_merge[lds]=2
        result_merge[(result_rgbh==2)]=2
        #result_merge[(result_base==2)]=2



        if use_msi_tree:
            msi_tree=cv2.imread(os.path.join(msitree_folder,image_name.replace('CLS','MSI')),0)
            msi_tree=convert_labels(msi_tree,params,toLasStandard=False)
            result_merge[(msi_tree==1)]=1
        ### water
        #lds=np.where((result_base==3)&(result_rgbh==3))
        #result_merge[lds]=3
        #result_merge[(result_rgbh==3)]=3
        #result_merge[(result_base==3)]=3
        result_merge[(result_base==3)]=3

        ### bridge
        result_merge[(result_base==4)]=4
        #result_merge[(result_base==4)]=4
        #result_merge[(result_rgbh==4)]=4
        #result_merge[(result_last==4)]=4

        ### tree
        #lds=np.where((result_base==1)&(result_rgbh==1))
        #result_merge[lds]=1
        result_merge[(result_last==1)]=1
        #result_merge[(result_base==1)]=1
        #result_merge[(result_rgbh==1)]=1
        

        result_merge=convert_labels(result_merge,params,toLasStandard=True)
        out_path=os.path.join(out_folder,site_name+'CLS.tif')
        tifffile.imsave(out_path,result_merge,compress=6)

def Merge_height(result_folders,label_folder, out_folder=''):
    if len(out_folder)<2:
        out_folder=result_folders[0]
    if os.path.exists(out_folder)==0:
        os.mkdir(out_folder)
    
    result_base=result_folders[0]
    glob_path=os.path.join(result_base,'*AGL.tif')
    files=glob.glob(glob_path)
    rgbh_folder=result_folders[1]
    if len(result_folders)>=3:
        use_msi_tree=True
        msitree_folder=result_folders[2]
    else:
        use_msi_tree=False

   # bridge_folder=result_folders[2]
    for img in files:
        image_name=os.path.split(img)[-1]
        site_name=image_name[0:-7]
        result_base=load_img(img)
        result_rgbh=load_img(os.path.join(rgbh_folder,image_name))
        label=cv2.imread(os.path.join(label_folder,image_name.replace('AGL','CLS')),0)
        label=convert_labels(label,params,toLasStandard=False)
        ###ground
        result_merge=np.zeros([result_base.shape[0],result_base.shape[1]], dtype = 'float32')
        result_merge[(label>=0)]=result_base[(label>=0)]

        ### roof
        result_merge[(label==2)]=result_base[(label==2)]
        wrong_roof_ids=np.where((label==2) & (result_base<1.5)& (result_base<result_rgbh))
        result_merge[wrong_roof_ids]=result_rgbh[wrong_roof_ids]
        
        ### tree
        result_merge[(label==1)]=result_base[(label==1)]#+result_rgbh[label==1])/2.0
        wrong_roof_ids=np.where((label==1) & (result_base<1)& (result_base<result_rgbh))
        result_merge[wrong_roof_ids]=result_rgbh[wrong_roof_ids]
        #lds=np.where((result_merge<result_rgbh)&(label==1))
        #result_merge[lds]=result_rgbh[lds]
        

        # if use_msi_tree:
        #     msi_tree=cv2.imread(os.path.join(msitree_folder,image_name.replace('CLS','MSI')),0)
        #     msi_tree=convert_labels(msi_tree,params,toLasStandard=False)
        # result_merge[(msi_tree==1)]=1

        ### water
        result_merge[(label==3)]=result_base[(label==3)]
        #result_merge[(result_base==3)]=result_base[(result_base==3)]

        ### bridge
        result_merge[(label==4)]=result_rgbh[(label==4)]
        #lds=np.where((result_merge<result_base)&(label==4))
        #result_merge[lds]=result_base[lds]
        

        result_merge=convert_labels(result_merge,params,toLasStandard=True)
        out_path=os.path.join(out_folder,site_name+'AGL.tif')
        tifffile.imsave(out_path,result_merge,compress=6)

def AGLScale_change(agl_folder, label_folder,out_folder):
    if os.path.exists(out_folder)==0:
        os.mkdir(out_folder)
    glob_path=os.path.join(agl_folder,'*AGL.tif')
    files=glob.glob(glob_path)
    for img in files:
        image_name=os.path.split(img)[-1]
        agl=load_img(img)
        label=load_img(os.path.join(label_folder,image_name.replace('AGL','CLS')))
        ###for roof
        scaled_map=-0.0001293*np.square(agl) + 1.168 *agl - 0.697
        agl[label==5]=scaled_map[label==5]
        ###for bridge
        scaled_map=np.square(agl)*0.003896  + 1.022 *agl - 0.0901
        agl[label==17]=scaled_map[label==17]    
        ###for tree
        #scaled_map=np.square(agl)*0.003896  + 1.022  *agl - 0.0901
        #agl[label==17]=scaled_map[label==17] 
        tifffile.imsave(os.path.join(out_folder,image_name),agl,compress=6)

def merge_all_bridge(all_result_folders,label_folder,merged_folder):
    num_test=len(all_result_folders)
    glob_path=os.path.join(label_folder,'*CLS.tif')
    files=glob.glob(glob_path)
    for label_img in files:
        label=load_img(label_img)
        agl_map=np.zeros([label.shape[0],label.shape[1]], dtype = 'float32')  
        image_name=os.path.split(label_img)[-1]
        for i in range(len(all_result_folders)):
            folder=all_result_folders[i]
            agl=load_img(os.path.join(folder,image_name.replace('CLS','AGL')))
            ids=np.where((label==17) & (agl_map<agl))
            agl_map[ids]=agl[ids]
        tifffile.imsave(os.path.join(merged_folder,image_name.replace('CLS','AGL')),agl_map,compress=6)





if __name__ == '__main__':
    # all_result_folders={}
    # all_result_folders[0]='../data/validate/Track1_AGL_merge_final222'
    # all_result_folders[1]='../data/validate/track1-unet_rgbc_h-moreabs-finall-2'
    # all_result_folders[2]='../data/validate/track1-rgbc_h-evenloss_lin-finall-2'
    # all_result_folders[3]='../data/validate/track1-rgbc_h-moreabs-finall-1'
    # all_result_folders[4]='../data/validate/track1-rgbc_h-moreabs-finall-1'
    # all_result_folders[5]='../data/validate/track1-rgbc_h-evenloss_lin-20190322e011-test'
    # all_result_folders[6]='../data/validate/Track1_Merge_h_190322_lin'
    # all_result_folders[7]='../data/validate/track1-rgbc_h-evenloss_lin-20190322e05-test'
    # all_result_folders[8]='../data/validate/track1-rgbc_h-evenloss_lin-20190322e07-test'
    # all_result_folders[9]='../data/validate/track1-unet_rgb_h-evenloss-lr-4-20190322e02-test'
    # all_result_folders[10]='../data/validate/track1-unet_rgb_h-evenloss-retrain-20190322e06-test'
    # all_result_folders[11]='../data/validate/Track1_AGL_merge_final'
    # all_result_folders[12]='../data/validate/Track1_Merge_h_190321_t1'
    # all_result_folders[13]='../data/validate/Track1_Merge_h_190321_t1'
    # all_result_folders[14]='../data/validate/Track1-rgbc_h-ganclass-0320e05-moreabs_finalclass'


    # label_folder='C:/TrainData/Track1/Test-Track1/pred_class'
    # merged_folder='../data/validate/Track1_agl_bridge_final_mrege'




    # merge_all_bridge(all_result_folders,label_folder,merged_folder)
    # baseline_7771_building='C:/Users/VS 02/Downloads/track1_submit20190318_t1'
    # baseline_7798_building='C:/Users/VS 02/Downloads/track1_submit20190321_t1'
    # baseline_7781='C:/Users/VS 02/Downloads/track1_submit20190314_t2'
    # water_bridge='../data/validate/Track1_class_final_water_bridge'
    # building='../data/validate/Track1_class_final_building'
    # tree='../data/validate/Track1_class_final_tree222'
    # new_resutls='../data/validate/track1-rgb_c-morenew-new-20190321e13-test'
    
    # rgb_c_folder='../data/validate/Track1_class_merge_building'#D:/grss2019/data/validate/Track1-rgb_c-20190311e27-new70000'
    # rgbh_folder='../data/validate/Track1_class_merge_tree2'#D:/grss2019/data/validate/Track1-rgbh_c-gan20190311e21-new70000-newdsm'
    # all_result_folders={}
    # all_result_folders[0]=water_bridge
    # all_result_folders[1]=building
    # all_result_folders[2]=tree
    # merged_folder='../data/validate/Track1_class_final_merge222'

    # Merge_all_results_with_baseline(all_result_folders,merged_folder)

#########################
    label_folder='C:/TrainData/Track1/Test-Track1/pred_class'
    #best_tree='../data/validate/track1-unet_rgb_h-evenloss-retrain-20190322e06-test'#D:/grss2019/data/validate/Track1-rgb_c-20190311e27-new70000'
    #best_building='../data/validate/track1-unet_rgb_h-evenloss-lr-4-20190322e02-test'#D:/grss2019/data/validate/Track1-rgbh_c-gan20190311e21-new70000-newdsm'
    best_bridge='../data/validate/Track1_agl_bridge_final_mrege'
    best_tree_building='../data/validate/Track1_AGL_merge_building_tree'
    all_result_folders={}
    all_result_folders[0]=best_tree_building
    all_result_folders[1]=best_bridge
    #all_result_folders[2]=best_building

#       all_result_folders[2]=out_folder
    merged_folder='../data/validate/Track1_AGL_merge_final3333'
    Merge_height(all_result_folders,label_folder,merged_folder)



#     # label_folder='../data/validate/Track1_class_merge_190321_t222'
#     # agl_folder='../data/validate/Track1-rgb_h-0320e01-evenloss_nomalvalue'
#     # out_folder='../data/validate/Track1-rgb_h-0320e01-evenloss_nomalvalue_scaled'
#     # AGLScale_change(agl_folder, label_folder,out_folder)

#     from grss_data import constrain_height
#     dsmout_folder='../data/validate/Track1_AGL_merge_final_constrain2222/'
#     constrain_height(merged_folder,label_folder,dsmout_folder)    

