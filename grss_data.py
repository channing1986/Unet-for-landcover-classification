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
import matplotlib.pyplot as plt
import params
import tifffile
from dataFunctions import convert_labels,load_img
from DataAugment import normalize_image_to_path,augment_online,convertLas2Train
import multiprocessing as mp
from dataFunctions import image_augmentation

TRAIN_task='CLS'
EXTRA_format='h'
EXTRA_data=False
NUM_class=3
is_MSI=False


def get_batch_inds(batch_size, idx, N,predict=False):
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
        if idx1 >= N:
            idx1 = N
            if predict==False:

                idx0 = idx1 - batch_size
            toProcess = False
        batchInds.append(idx[idx0:idx1])
        idx0 = idx1

    return batchInds

def GetPredictData(ortho_path, channels,extra_path='',path_size=(256,256),overlap_ratio=0.5,convertLab=False,normalize_dsm=0,img_ioa=[],resize2size=[]):
    from scipy import ndimage
    
    if len(channels)>3:
        extra_input=True
        if channels[-1]=='c':
            EXTRA_format='CLS'
        elif channels[-1]=='h':
            EXTRA_format='AGL'
    else:
        extra_input=False
    img_aug=[]
    from DataAugment import normalize_image_to_path,getAuger_p_p
    #label_input=False
    [imgs,dsms,image_size]=normalize_image_to_path(ortho_path,extra_path,path_size,overlap_ratio,extra_input=extra_input,work_region=img_ioa,convertLab=convertLab,resize2size=resize2size,pad_edge=1,normalize_dsm=False, image_order=1)
    contrast_range_2=(0.9,1.1)
    for i in range(len(imgs)):
        img_data=imgs[i]
        #img_data = img_data.transpose(1,2,0)
        img_data=img_data.astype(np.float32)
        if channels[:3]=='rgb':                    
            img_data=img_data/125.0-1
        elif channels[:3]=='msi':
            img_data=img_data          
        if extra_input:
            if EXTRA_format=='AGL':
                extra_dat = dsms[i].astype(np.float32)/20-1
               #extra_dat = dsms[i].astype(np.float32)-0.5 ## for track 2
            elif EXTRA_format=='CLS':
                extra_dat = dsms[i].astype(np.float32)/2-1
            extra_dat = extra_dat[:,:,np.newaxis]
            img_data=np.concatenate((img_data, extra_dat), axis=-1)
        for a_id in range (1):
            img_a=np.rot90(img_data,4-a_id)
            #img_a2=ndimage.rotate(img_data,a_id*-90)
            if 0:
                plt.subplot(131) #用于显示多个子图121代表行、列、位置
                plt.imshow(img_data)
                plt.title('org')
                plt.subplot(132)
                plt.imshow(img_a)
                plt.title('rote90') #添加标题
                plt.subplot(133)
                plt.imshow(img_a)
                plt.title('rote90_2') #添加标题
                plt.show()
           # auger=getAuger_p_p(a_id)
            #img_a=auger.augment_image(img_data)
            img_aug.append(img_a)
        # for a_id in range (4):
        #     auger=getAuger_p_p(a_id)
        #     img_a=auger.augment_image(rgbh)
        #     img_aug.append(img_a)
    return img_aug,image_size


 
def input_generator(img_files, label_files, batch_size, extra_files=[],net_name='',num_category=5):

    global TRAIN_task
    global EXTRA_data
    global EXTRA_format
    global NUM_class
    global is_MSI
    NUM_class=num_category
    channels=net_name.split('_')[1]
    if net_name[-1]=='c':
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

    #executor = ProcessPoolExecutor(max_workers=3)

    while True:
        for inds in batchInds:
            img_batch = [img_files[ind] for ind in inds]
            label_batch = [label_files[ind] for ind in inds]
            if len(channels)>3:
                extra_batch=[extra_files[ind] for ind in inds]
            else:
                extra_batch=['']
            imgdata, gts= load_cnn_batch(img_batch, label_batch,extra_batch,is_msi_data,num_category)#, executor)      
            
            if 0:
                import matplotlib.pyplot as plt 
                plt.subplot(221) #用于显示多个子图121代表行、列、位置
                plt.imshow(imgdata[0,:,:])
                plt.title('org')
                plt.subplot(222)
                plt.imshow(imgdata[1,:,:])
                plt.title('background') #添加标题
                plt.subplot(223)
                plt.imshow(gts[0,:,:,1])
                plt.title('dsm') #添加标题
                plt.subplot(224)
                plt.imshow(gts[1,:,:,1])
                plt.title('roof') #添加标题
                plt.show()
            yield (imgdata, gts)
            #return (imgdata, gts)

def load_cnn_batch(img_batch, label_batch, extra_batch, is_msi_data,num_class,executor=''):
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
        if len(extra_batch)>3:
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

    y_train=np.array(labels, np.float32)
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
    extra_file=inputDict['extra']
    num_class=inputDict['num_class']
    inputs=[]
    labels=[]
    img_data=load_img(rgb_file)
    if len(img_data.shape)<3:
        img_data=cv2.cvtColor(img_data,cv2.COLOR_GRAY2RGB)
    label_data=load_img(gts_file)
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
            
    currLabel=np.array(label_data,np.float)
    label_data = to_categorical(currLabel, num_classes=int(num_class))
    Remove_last_label=False
    if Remove_last_label:
        label_data =label_data[:,:,0:-1]
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
    # inputs.append(img_data)
    # labels.append(label_data)   
    return inputs, labels
def input_generator_mp(img_files, label_files, batch_size, extra_files=[],net_name='',num_category=5):
    """

    """
    pool=mp.Pool(processes = 6);
    N = len(img_files) #total number of images
    idx = np.random.permutation(N) #shuffle the order
    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 
            
    NUM_class=num_category
    channels=net_name.split('_')[1]
    if net_name[-1]=='c':
        TRAIN_task='CLS'
    else:
        TRAIN_task='AGL'
    if channels[0:3]=='msi':
        is_MSI=True
        is_msi_data=True
    if len(channels)>=4:
        EXTRA_data=True
        if channels[-1]=='c':
            EXTRA_format='c'
        else:
            EXTRA_format='h'

    while True:
        for inds in batchInds:
            imgdata=[]
            labels=[]
            img_batch = [img_files[ind] for ind in inds]
            label_batch = [label_files[ind] for ind in inds]
            if len(channels)>3:
                extra_batch=[extra_files[ind] for ind in inds]
            else:
                extra_batch=['']
            img_label=zip(img_batch,label_batch,extra_batch)
            #res=load_img_label(img_batch[0],label_batch[0])######
            res=pool.starmap(load_img_label_seg,img_label)
            for img,lable in res:
                imgdata.append(img)
                labels.append(lable)

            labels = np.array(labels, np.float32)
            imgdata = np.array(imgdata, np.float32)

    #return imgdata,labels

            if EXTRA_data:
                if not is_msi_data:
                    x_train[:,:,:,0:-1]=x_train[:,:,:,0:-1]/125.0-1
                if EXTRA_format=='h':
                    x_train[:,:,:,-1]=x_train[:,:,:,-1]/20.0-1
                elif EXTRA_format=='c':
                    x_train[:,:,:,-1]=x_train[:,:,:,-1]/2.0-1
            else:
                if not is_msi_data:
                    x_train=x_train/125.0-1
            #yield (imgdata, labels)
            return imgdata,labels

def load_img_label_seg(rgb_file,gts_file,extra_file):
    inputs=[]
    labels=[]
    img_data=load_img(rgb_file)
    label_data=load_img(gts_file)

    img_data=img_data.astype(np.float)
    if len(extra_file)>3:
        #print('sfd')
        extra_data=load_img(extra_file)
        if extra_file[-9:-6]=='AGL':
            extra_data[np.isnan(extra_data)]=params.IGNORE_VALUE
        extra_data = extra_data.astype(np.float)
        extra_data = extra_data[:,:,np.newaxis]
        img_data=np.concatenate((img_data, extra_data), axis=-1)
    if gts_file[-9:-6]=='AGL':
        label_data[np.isnan(label_data)]=params.IGNORE_VALUE
    elif gts_file[-9:-6]=='CLS':
        label_data = label_data.astype(np.float)
        currLabel=np.array(label_data,np.float)
        currLabel = to_categorical(currLabel, num_classes=NUM_class+1)
        label_data =currLabel[:,:,0:-1]
    # label_data[label_data==1] = 0
    # label_data[label_data==2] = 1
    # label_data[label_data==3] = 0
    # label_data[label_data==4] = 2
    # label_data[label_data==5] = 0
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
            

    imageMedium,labelMedium = image_augmentation(img_data, label_data)
    inputs.append(img_data)
    labels.append(label_data)    
    inputs.append(imageMedium)
    labels.append(labelMedium)
    return inputs, labels


    #return rgb, gts
def constrain_height(results_folder,label_folder,out_folder):

    if os.path.exists(out_folder)==0:
        os.mkdir(out_folder)
    
    glob_path=os.path.join(results_folder,'*AGL.tif')
    files=glob.glob(glob_path)
    height_range=[[-0.49,0.49],[0,31],[0,80],[-0.49,0.49],[0,100]]
    for img in files:
        
        image_name=os.path.split(img)[-1]
        site_name=image_name[0:-7]
        result_my=load_img(img)
        #result_my=convert_labels(result_my,params,toLasStandard=False)
        label_file=image_name.replace('AGL','CLS')
        label=load_img(os.path.join(label_folder,label_file))
        label=convert_labels(label,params,toLasStandard=False)

        for i in range(5):
            lds=np.where((result_my<height_range[i][0])&(label==i))
            result_my[lds]=height_range[i][0]

            lds=np.where((result_my>height_range[i][1])&(label==i))
            result_my[lds]=height_range[i][1]

        #     result_merge=np.zeros([label.shape[0],label.shape[1]], dtype = 'float32')
            
        # result_my[label==0]=0.0
        # result_my[label==3]=0.0

        out_path=os.path.join(out_folder,image_name)
        tifffile.imsave(out_path,result_my,compress=6)
def calculate_confusion_matrix(yp_folder,yt_folder):
    from sklearn.metrics import confusion_matrix

    glob_path=os.path.join(yp_folder,'*CLS.tif')
    files=glob.glob(glob_path)
    All_confus_matrix=np.zeros((6,6))
    for img in files:
        yp=load_img(img)
        yp=convert_labels(yp,params,toLasStandard=True)
        image_name=os.path.split(img)[-1]
        yt=load_img(os.path.join(yt_folder,image_name))
        yp = np.array(yp)
        yt = np.array(yt)
        yp_num=yp.flatten()
        
        yt_num=yt.flatten()
        c_matrix=confusion_matrix(yt_num,yp_num,labels=[2,5,6,9,17,65])
        All_confus_matrix=All_confus_matrix+c_matrix
    np.savetxt('confusion_matraix.txt', All_confus_matrix)   # X is an array
def load_all_data_files_balanced_patches_singel(data_folder,data_txt,label_folder='',vali_ratio=0.1,net_name='unet_rgbc_c'):
    if len(label_folder)<3:
        class_folder='label_patch'
    else:
        class_folder=label_folder
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
        is_extrac_data=False
    val_num=300
    imgs = []
    gts=[]
    extras=[]

    imgs_v=[]
    gts_v=[]
    extras_v=[]
    label_list=os.path.join(data_folder,'label_list.txt')
    img_files=[]
    fp = open(label_list)
    lines = fp.readlines()
    fp.close()
    for line in lines:
        line = line.strip('/n')
        img_files.append(line)
    ids=[]
    train_ids=[]
    val_ids=[]
    fp = open(os.path.join(data_folder,data_txt))
    lines = fp.readlines()
    fp.close()

    for line in lines:
        line = line.strip('/n')
        ids.append(int(line))
    train_ids=ids[0:int(len(ids)*(1-vali_ratio))]
    val_ids=ids[int(len(ids)*(1-vali_ratio)):]


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
    if is_extrac_data:
        return imgs,gts,extras,  imgs_v, gts_v, extras_v
    else:
        return imgs, gts, imgs_v,  gts_v

def load_all_data_files_balanced_patches(data_folder,label_folder='',text_files='',vali_ratio=0.1,max_samples=-1,net_name='unet_rgb_c'):
    
    if not label_folder:
        class_folder='label_patch'
    else:
        class_folder=label_folder
    
    dsm_folder='dsm_patch'
    img_folder='img_patch'
    COLOR='RGB'
    channels=net_name.split('_')[1]
    if channels[0:3]=='msi':        
        img_folder='msi_treewater_patch'
        class_folder='label_patch_treewater'
        #COLOR='MSI'
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
    
    if len(channels)==4:
        is_extrac_data=True
    if channels[-1]=='c':
        extrac_folder=class_folder
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

    imgs_v=[]
    gts_v=[]
    extras_v=[]
    label_list=os.path.join(data_folder,'label_list.txt')
    bad_list_path=os.path.join(data_folder,'bad_list.txt')
    img_files=[]
    fp = open(label_list)
    lines = fp.readlines()
    fp.close()
    for line in lines:
        line = line.strip('\n')
        img_files.append(line)

    bad_list=[]
    fp = open(bad_list_path)
    lines = fp.readlines()
    fp.close()
    for line in lines:
        line = line.strip('\n')
        bad_list.append(int(line))

    bad_list=set(bad_list)
    if len(text_files)<1:
        text_files=[os.path.join(data_folder,'ground_list.txt'),
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
    if max_samples<1:
        max_samples=max(clasee_samples)
    min_samples=min(clasee_samples)
    record_sampels=0
    #extrac the validation data first
    val_num=int(3000*vali_ratio)
    val_ids=[]
    for i in range(len(text_files)):
        class_ids=all_ids[i]
        idx = np.random.permutation(len(class_ids))
        ids=idx[0:val_num]
        val_ids.extend(class_ids[ind] for ind in ids)
    train_ids=[]
    val_ids=set(val_ids)-bad_list
    batch_star=[0,0,0,0,0]
    while record_sampels<max_samples:
        batch_ids=[]
        
        for i in range(len(text_files)):
            class_ids=all_ids[i]
            batch_end=batch_star[i]+min_samples
            if batch_end>=clasee_samples[i]:
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
        batch_ids=set(batch_ids)-bad_list
        batch_ids=batch_ids-val_ids
        train_ids.extend(batch_ids)
        record_sampels=record_sampels+min_samples

    for id in train_ids:
        line=img_files[id-1]
        if task=='AGL':
            gts.append(os.path.join(data_folder,dsm_folder,line.replace('CLS',task)))
        elif task=='CLS':
            gts.append(os.path.join(data_folder,class_folder,line))
        img_path=line.replace('CLS',COLOR)
        imgs.append(os.path.join(data_folder,img_folder,img_path))
        extra_path=line.replace('CLS',extrac_format)
        extras.append(os.path.join(data_folder,extrac_folder,extra_path))
    for id in val_ids:
        line=img_files[id-1]
        if task=='AGL':
            gts_v.append(os.path.join(data_folder,dsm_folder,line.replace('CLS',task)))
        elif task=='CLS':
            gts_v.append(os.path.join(data_folder,class_folder,line))
        img_path=line.replace('CLS',COLOR)
        imgs_v.append(os.path.join(data_folder,img_folder,img_path))
        extra_path=line.replace('CLS',extrac_format)
        extras_v.append(os.path.join(data_folder,extrac_folder,extra_path))
    if is_extrac_data:
        return imgs,gts,extras,  imgs_v, gts_v, extras_v
    else:
        return imgs, gts, imgs_v,  gts_v
def load_all_data_files(data_folder,vali_ratio=0.1):
    IMG_FOLDER='img_patch'
    LABEL_FOLDER='label_patch'
    IMG_TXT='img_list.txt'
    LABEL_TXT='label_list.txt'
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


def load_all_data_test(data_folder, extension='.tif'):
    pathes=[]
    #img_folder=os.path.join(data_folder,'epipolar image')
    #img_folder=os.path.join(data_folder,'imgs')
    img_files = glob.glob(os.path.join(data_folder, '*'+extension))
    for imgPath in img_files:
        imageName = os.path.split(imgPath)[-1]
        pathes.append(imgPath)
    return pathes

def GetSmallTreeLabel(label_folder,out_folder):
    from skimage import data, util, color,measure
    import matplotlib.pyplot as plt
    #from skimage.measure import label as m_label
    if os.path.exists(out_folder)==0:
        os.mkdir(out_folder)
    MAX_NUMBER=4000
    glob_path=os.path.join(label_folder,'*CLS*.tif')
    files=glob.glob(glob_path)

    for img in files:
        
        image_name=os.path.split(img)[-1]
        site_name=image_name[0:-7]
        label_data=load_img(img)
        bw_img=np.zeros(label_data.shape[:2],np.uint8)
        bw_img[label_data==1]=1

        labeled_img, num = measure.label(bw_img, neighbors=4, background=0, return_num=True)

        for i in range(1, num): # 这里从1开始，防止将背景设置为最大连通域
            totol_num=np.sum(labeled_img == i)
            if totol_num >= MAX_NUMBER:
                bw_img[labeled_img == i] = 3

        # dst = color.label2rgb(bw_img)
        # plt.imshow(dst)
        # plt.show()
        out_path=os.path.join(out_folder,image_name)
        tifffile.imsave(out_path,bw_img,compress=6)
def GenerateTreeData(label_folder):
    output_folder='C:/TrainData/Track1/train/patch_512/label_patch_tree'
    GetSmallTreeLabel(label_folder,output_folder)

def GenerateTreeWaterMSIdata(out_folder):
    if os.path.exists(out_folder)==0:
        os.mkdir(out_folder)



def GenerateTreeWaterLabelandData(data_folder,label_path,labelout_path,dataout_path):
    if os.path.exists(labelout_path)==0:
        os.mkdir(labelout_path)
    if os.path.exists(dataout_path)==0:
        os.mkdir(dataout_path)
    label_folder=os.path.join(data_folder,'label_patch')
    msi_folder=os.path.join(data_folder,'msi_patch')
    rgb_folder=os.path.join(data_folder,'img_patch')
    label_list=os.path.join(data_folder,'label_list.txt')
    image_names=[]
    fp = open(label_list)
    lines = fp.readlines()
    fp.close()
    for line in lines:
        line = line.strip('\n')
        image_names.append(line)

    tree_list_file=os.path.join(data_folder,'tree_list.txt')
    water_list_file=os.path.join(data_folder,'water_list.txt')
    text_files=[tree_list_file,water_list_file]

    all_ids=[]
    if os.path.exists(tree_list_file) and os.path.exists(water_list_file):
        for i in range(len(text_files)):
            fp = open(text_files[i])
            lines = fp.readlines()
            fp.close()
            ids=[]
            for line in lines:
                line = line.strip('\n')
                ids.append(int(line))
            all_ids.extend(ids)
        all_ids=set(all_ids)


        for id in all_ids:
            i=id-1
            label=load_img(os.path.join(label_path,image_names[i]))
            bw_img=np.zeros(label.shape[:2],np.uint8)
            bw_img[label==1]=1
            bw_img[label==3]=2
            bw_img[label==5]=3
            out_path=os.path.join(labelout_path,image_names[i])
            tifffile.imsave(out_path,bw_img,compress=6 )

            msi=load_img(os.path.join(msi_folder,image_names[i].replace('CLS','MSI')))
            avg_r=(msi[:,:,6]+msi[:,:,7])/2
            ndvi=(avg_r-msi[:,:,4])/(avg_r+msi[:,:,4])
            avg_w=(msi[:,:,0]+msi[:,:,1])/2
            ndwi1=(avg_w-msi[:,:,6])/(avg_w+msi[:,:,6])

            #ndwi2=(avg_w-msi[:,:,7])/(avg_w+msi[:,:,7])

            rgb=load_img(os.path.join(rgb_folder,image_names[i].replace('CLS','RGB')))
            itens=np.mean(rgb,axis=-1)/125-1
            data=np.zeros((label.shape[0],label.shape[1],3),np.float32)
            data[:,:,0]=itens#ndwi2#itens
            find_nan = np.where(ndvi == ndvi, 1, 0)
            if np.min(find_nan)==0:
                print('NAN in image:',image_names[i])
            # if len(np.isinf(ndvi))>0:
            #     ccc=0
            ndvi[np.isnan(ndvi)]=0
            ndvi[np.isinf(ndvi)]=0
            ndwi1[np.isnan(ndwi1)]=0
            ndwi1[np.isinf(ndwi1)]=0
            data[:,:,1]=ndvi
            data[:,:,2]=ndwi1
            out_path=os.path.join(dataout_path,image_names[i].replace('CLS','RGB'))
            tifffile.imsave(out_path,data,compress=6 )            

def convertMSI2watertreedata(msi_path):
    if 1:
        if 1:
            data_folder=os.path.split(msi_path)[-2]
            image_name=os.path.split(msi_path)[-1]
            msi=load_img(msi_path)
            avg_r=(msi[:,:,6]+msi[:,:,7])/2
            ndvi=(avg_r-msi[:,:,4])/(avg_r+msi[:,:,4])
            avg_w=(msi[:,:,0]+msi[:,:,1])/2
            ndwi1=(avg_w-msi[:,:,6])/(avg_w+msi[:,:,6])

            #ndwi2=(avg_w-msi[:,:,7])/(avg_w+msi[:,:,7])

            rgb=load_img(os.path.join(data_folder,image_name.replace('MSI','RGB')))
            itens=np.mean(rgb,axis=-1)/125-1
            data=np.zeros((msi.shape[0],msi.shape[1],3),np.float32)
            data[:,:,0]=itens#ndwi2#itens
            data[:,:,1]=ndvi
            data[:,:,2]=ndwi1
    return data


def GenerateTreeWaterSMIdata(data_folder,label_folder,dataout_folder,labelout_folder):
    label_path=os.path.join(data_folder,label_folder)
    labelout_path=os.path.join(data_folder,labelout_folder)
    dataout_path=os.path.join(data_folder,dataout_folder)
    GenerateTreeWaterLabelandData(data_folder,label_path,labelout_path,dataout_path)

def GetInputData(img_path,extra_path,work_region=(),convertLab=False,resize2size=(),pad_edge=1,normalize_dsm=False):
    from grss_data import convertMSI2watertreedata
    if img_path[-7:]=='MSI.tif':
        is_msi=True
    elif img_path[-7:]=='RGB.tif':
        is_msi=False
    else:
        print('wrong input img_path:',img_path)

    if is_msi:
        img=convertMSI2watertreedata(img_path)
    else:
        img=load_img(img_path)
    if len(extra_path)>3:
        extra_input=True
        extra_data=load_img(extra_path)
    else:
        extra_input=False
        extra_data=[]
    if extra_input:
        if convertLab:
            extra_data=convertLas2Train(extra_data, params.LABEL_MAPPING_LAS2TRAIN)
        elif normalize_dsm:
            nan_data=np.isnan(extra_data)
            extra_data[nan_data] = 99999
            min_t=extra_data.min()
            extra_data=extra_data-min_t
            extra_data[nan_data]=0

    if len(work_region)>0:
        img = img[:,work_region[0]:work_region[1],work_region[2]:work_region[3]]
        if extra_input:
            if (extra_data.shape[0]>work_region[1]-work_region[0] or extra_data.shape[1]>work_region[3]-work_region[2]):
                extra_data=extra_data[:,work_region[0]:work_region[1],work_region[2]:work_region[3]]

    if len(resize2size)>0:
        img=cv2.resize(img,resize2size)
        if extra_input:
            extra_data=cv2.resize(extra_data,resize2size,interpolation=cv2.INTER_NEAREST)
    return img, extra_data

def InputDataProcessing(img,channels,extradata): 
    img_aug=[]    
    if len(channels)>3:
        extra_input=True
        if channels[-1]=='c':
            EXTRA_format='CLS'
        elif channels[-1]=='h':
            EXTRA_format='AGL'
    else:
        extra_input=False  
    if channels[:3]=='rgb':                    
        img_data=img/125.0-1 
    if extra_input:       
        if EXTRA_format=='AGL':
                extra_dat = extradata.astype(np.float32)/20.-1   ## for track3 is /50.-1,   /20.0-1
        elif EXTRA_format=='CLS':
            extra_dat = extradata.astype(np.float32)/2-1
        extra_dat = extra_dat[:,:,np.newaxis]
        img_data=np.concatenate((img_data, extra_dat), axis=-1)
    for a_id in range (4):
        img_a=np.rot90(img_data,4-a_id)
            #img_a2=ndimage.rotate(img_data,a_id*-90)
        if 0:
            plt.subplot(131) #用于显示多个子图121代表行、列、位置
            plt.imshow(img_data)
            plt.title('org')
            plt.subplot(132)
            plt.imshow(img_a)
            plt.title('rote90') #添加标题
            plt.subplot(133)
            plt.imshow(img_a)
            plt.title('rote90_2') #添加标题
            plt.show()
           # auger=getAuger_p_p(a_id)
            #img_a=auger.augment_image(img_data)
        img_aug.append(img_a)
        # for a_id in range (4):
        #     auger=getAuger_p_p(a_id)
        #     img_a=auger.augment_image(rgbh)
        #     img_aug.append(img_a)
    return img_aug

def GetPredictData_new(ortho_path,channels='rgb',extra_path='',path_size=(),convertLab=False,overlap_ratio=0.5,resize2size=(),img_ioa=[]):
    from scipy import ndimage
    import matplotlib.pyplot as plt
    if len(channels)>3:
        extra_input=True
        if channels[-1]=='c':
            EXTRA_format='CLS'
        elif channels[-1]=='h':
            EXTRA_format='AGL'
    else:
        extra_input=False
    img_aug=[]
    from DataAugment import normalize_image_to_path,img2patches
    #label_input=False
    normalize_dsm=True  ## this is for track3.

    [data,extra_data]=GetInputData(ortho_path,extra_path,work_region=img_ioa,convertLab=convertLab,resize2size=resize2size,pad_edge=1,normalize_dsm=normalize_dsm)
    data_aug=InputDataProcessing(data,channels,extra_data)

    return data_aug,data.shape[0:3]
   


def VoteStrategeMapping(vote_map):
    #class_map=np.argmax(vote_map,axis=-1).astype('uint8')
    class_map=np.zeros((vote_map.shape[0],vote_map.shape[1],1))
    vote_softmax=(vote_map)/np.sum(vote_map,axis=-1,keepdims=True)
    ##add the most tree, then add the building, then add the water. 
    tree_vote=vote_softmax[:,:,1]
    roof_vote=vote_softmax[:,:,2]
    water_vote=vote_softmax[:,:,3]
    bridge_vote=vote_softmax[:,:,4]
    
    class_map[roof_vote>0.6]=2
    class_map[tree_vote>0.15]=1
    class_map[water_vote>0.30]=3
    class_map[bridge_vote>0.3]=4
    return class_map


    
def track3_Merge_temparal_results_new(result_folder,out_folder,track='track3',new_merge=True,if_convert_labels=False,offset=False,offset_folder=''):
    site_images=[]
    site_names=[]
    #back_folder='G:/programs/dfc2019-master/track1/data/validate/track2-beforMerge/'
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
        if track=='track3':
            site_name=image_name[0:7]
        elif track=='track1':
            site_name=image_name[0:11]
        else:
            site_name=image_name[0:-9]

        new_site=True
        for i in range(len(site_names)):
            if site_name==site_names[i]:
                new_site=False
                site_images[i].append(img)
        if new_site:
            site_names.append(site_name)
            site_images.append([img])
#              site_images[len(site_names)-1].append(img)
    NUM_CATEGORIES=params.NUM_CATEGORIES
    for m in range(len(site_names)):
        imgs=site_images[m]
        im=load_img(imgs[0])
        
        
        

        vote_map=np.zeros((im.shape[0],im.shape[1],NUM_CATEGORIES))
        for img_p in imgs:            
            im=load_img(img_p)

            #image_name=os.path.split(img_p)[-1]
            #pred_img=np.argmax(im,axis=-1).astype('uint8')
            #out_path=os.path.join(back_folder,image_name)
            #pred_img=convertLas2Train(pred_img, params.LABEL_MAPPING_TRAIN2LAS)
            #tifffile.imsave(out_path,pred_img,compress=6)
            if if_convert_labels:
                im=convertLas2Train(im, params.LABEL_MAPPING_LAS2TRAIN)
            im=to_categorical(im,NUM_CATEGORIES) ###different merge
            for i in range(vote_map.shape[-1]):
                vote_map[:,:,i]=vote_map[:,:,i]+im[:,:,i]
        #new_merge=False
        if new_merge:
            pred=VoteStrategeMapping(vote_map)
        else:
            pred=np.argmax(vote_map,axis=-1).astype('uint8')

        if offset:
            offset_file=os.path.join(offset_folder,site_names[m]+'_DSM.txt')
            offset = np.loadtxt(offset_file)
            offset=offset.astype('int')
            pred=pred[offset[1]:offset[1]+512,offset[0]:offset[0]+512]
        
        pred=convert_labels(pred,params,toLasStandard=True).astype('uint8')
        out_path=os.path.join(out_folder,site_names[m]+'_CLS.tif')
        tifffile.imsave(out_path,pred,compress=6)

if __name__ == '__main__':
    # data_folder='C:/TrainData/Track1/train/patch_512'
    # label_folder='label_patch'
    # dataout_folder='msi_treewater_patch'
    # labelout_folder='label_patch_treewater'
    # GenerateTreeWaterSMIdata(data_folder,label_folder,dataout_folder,labelout_folder)
    ###############
    resultFolder='../data/validate/CLS2Ortho'
    out_folder='../data/validate/track3-CLS2Ortho-0322-merged-more-building22/'
    track3_Merge_temparal_results_new(resultFolder, out_folder,track='track3',if_convert_labels=False)
    #########################
    # label_folder='C:/TrainData/Track1/Test-Track1/pred_class'
    # dsmout_folder='../data/validate/Track1-test-submit/'
    # constrain_height(dsmout_folder,label_folder,dsmout_folder)


