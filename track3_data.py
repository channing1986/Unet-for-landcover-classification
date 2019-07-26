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
class GRID:

    #读图像文件
    def read_img(self,filename):
        dataset=gdal.Open(filename)       #打开文件

        im_width = dataset.RasterXSize    #栅格矩阵的列数
        im_height = dataset.RasterYSize   #栅格矩阵的行数

        im_geotrans = dataset.GetGeoTransform()  #仿射矩阵
        im_proj = dataset.GetProjection() #地图投影信息
        im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵

        del dataset 
        return im_proj,im_geotrans,im_data

    #写文件，以写成tif为例
    def write_img(self,filename,im_proj='',im_geotrans='',im_data=''):
        #gdal数据类型包括
        #gdal.GDT_Byte, 
        #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        #gdal.GDT_Float32, gdal.GDT_Float64

        #判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        #判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape 

        #创建文件
        driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        if im_geotrans:
            dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        if im_proj:
            dataset.SetProjection(im_proj)                    #写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(np.squeeze(im_data))  #写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])

        del dataset

gdal_reader=GRID()

Dataset_path='C:/TrainData/Track3/Train/patch_512/'
img_folder='img_patch/'
label_folder='label_patch/'
dsm_folder='dsm_patch/'
num_workers=6
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
def load_all_data_files(data_folder):
    balanced_sample_number=3200
    imgs = []
    gts=[]
    dsm=[]
    imgs_v=[]
    gts_v=[]
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

                gts.append(os.path.join(data_folder,label_folder,line))
                img_path=line.replace('CLS','ortho')
                imgs.append(os.path.join(data_folder,img_folder,img_path))
                dsm_path=line.replace('CLS','DSM')
                dsm.append(os.path.join(data_folder,dsm_folder,dsm_path))
            else:
                gts_v.append(os.path.join(data_folder,label_folder,line))
                img_path=line.replace('CLS','ortho')
                imgs_v.append(os.path.join(data_folder,img_folder,img_path))
                dsm_path=line.replace('CLS','DSM')
                dsm_v.append(os.path.join(data_folder,dsm_folder,dsm_path)) 
            count=count+1               
    
    #return imgs[val_range:len(imgs)], gts[val_range:len(imgs)], imgs[0:val_range], gts[0:val_range]
    return imgs, dsm,gts, imgs_v, dsm_v, gts_v
def load_all_data_test(ortho_txt,dsm_txt):
    orthos=[]
    dsms=[]
    fp = open(ortho_txt)
    lines = fp.readlines()
    for line in lines:
            line = line.strip('\n')           
            orthos.append(line)
    fp.close()
    fp = open(dsm_txt)
    lines = fp.readlines()
    for line in lines:
            line = line.strip('\n')           
            dsms.append(line)
    fp.close()
    if len(orthos)!=len(dsms):
        return [], []
    return orthos,dsms


def load_all_image_RGBH(dset, image_data_format):
    import os
    from os import walk
    import string
    imgs = []
    gts=[]
    dsm=[]
    image_folder=dset+'img_out/'
    label_folder=dset+'gts_out/'
    dsm_folder=dset+'dsm_out/'

    for (dirpath, dirnames, filenames) in walk(image_folder):
        for img in filenames:
            img_format=img[-3:]
            if img_format==image_data_format:
               if os.path.exists(label_folder+img[:-4]+'_label.tif'):
                    if os.path.exists(dsm_folder+img[:-4]+'_dsm.tif'):               
                       gts.append(label_folder+img[:-4]+'_label.tif')
                       imgs.append(image_folder+img)
                       dsm.append(dsm_folder+img[:-4]+'_dsm.tif')
    val_range=int(0.2*len(imgs))
    return imgs[val_range:len(imgs)], dsm[val_range:len(imgs)],gts[val_range:len(imgs)], imgs[0:val_range], dsm[0:val_range], gts[0:val_range]
    #return imgs[5000:len(imgs)], dsm[5000:len(imgs)],gts[5000:len(imgs)], imgs[0:5000],dsm[0:5000], gts[0:5000]
    #return imgs[0:100], dsm[0:100],gts[0:100], imgs[100:200],dsm[100:200], gts[100:200]
def input_generator_RGBH_p(img_files,dsm_files, label_files, batch_size):
    """
    """

    N = len(img_files) #total number of images

    #idx = np.random.permutation(N) #shuffle the order
    idx = range(N)
    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    executor = ProcessPoolExecutor(max_workers=num_workers)

    while True:
        for inds in batchInds:
            img_batch = [img_files[ind] for ind in inds]
            dsm_batch = [dsm_files[ind] for ind in inds]
            label_batch = [label_files[ind] for ind in inds]
            imgdata, gts= load_cnn_batch_RGBH(img_batch, dsm_batch,label_batch,  executor)      
            yield (imgdata, gts)
def input_generator_RGBH(img_files,dsm_files, label_files, batch_size):
    """
    """

    N = len(img_files) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    executor = ProcessPoolExecutor(max_workers=num_workers)

    while True:
        for inds in batchInds:
            img_batch = [img_files[ind] for ind in inds]
            dsm_batch = [dsm_files[ind] for ind in inds]
            label_batch = [label_files[ind] for ind in inds]
            imgdata, gts= load_cnn_batch_RGBH(img_batch, dsm_batch,label_batch,  executor)      
            yield (imgdata, gts)
          #  return (imgdata, gts)

def load_cnn_batch_RGBH(img_batch,dsm_batch, label_batch,  executor):
    """
    """
    results=[]
    imgdata=[]
    labels=[]
    futures = []
    for i in range(0, len(img_batch)):
        currInput = {}
        currInput['gts'] =label_batch[i]
        currInput['dsm'] =dsm_batch[i]
        currInput['rgb'] = img_batch[i]

        futures.append(executor.submit(_load_batch_helper_RGBH, currInput))
 #       result=_load_batch_helper_RGBH(currInput)

        

    results = [future.result() for future in futures]
       # results.append( _load_batch_helper(currInput))
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
        x_train[:,:,:,0:-1]=x_train[:,:,:,0:-1]/256.0-1
        x_train[:,:,:,-1]=x_train[:,:,:,-1]/50.0-1

    return x_train, y_train


def _load_batch_helper_RGBH(inputDict):
    """
    Helper for load_cnn_batch that actually loads imagery and supports parallel processing
    :param inputDict: dict containing the data and metadataStats that will be used to load imagery
    :return currOutput: dict with image data, metadata, and the associated label
    """
    #print("fsf")
    #return
    rgb_file = inputDict['rgb']
    dsm_file=inputDict['dsm']
    gts_file = inputDict['gts']


    inputs=[]
    labels=[]
    proj2,geotrans2,img_data=gdal_reader.read_img(rgb_file)
    proj1,geotrans1,nDSM_data=gdal_reader.read_img(gts_file)
    proj,geotrans,DSM_data=gdal_reader.read_img(dsm_file)

    
    img_data=img_data.astype(np.float)
    img_data = img_data.transpose(1,2,0)
    nDSM_data = nDSM_data.astype(np.float)
    DSM_data = DSM_data.astype(np.float)
    currLabel = to_categorical(nDSM_data, num_classes=5+1)
    currLabel =currLabel[:,:,0:-1]
    DSM_data = DSM_data[:,:,np.newaxis]
 #   nDSM_data = nDSM_data[:,:,np.newaxis]
    rgbh=np.concatenate((img_data, DSM_data), axis=-1)

    from dataFunctions import image_augmentation
    imageMedium,labelMedium = image_augmentation(rgbh, currLabel)
    inputs.append(rgbh)
    inputs.append(imageMedium)
    labels.append(currLabel)
    labels.append(labelMedium)
    return inputs, labels


def input_generator(img_pathes,label_pathes,batch_size):
    """
    """

    N = len(img_pathes) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    executor = ProcessPoolExecutor(max_workers=num_workers)

    while True:
        for inds in batchInds:
            img_batch = [img_pathes[ind] for ind in inds]
            label_batch = [label_pathes[ind] for ind in inds]
            imgdata, gts= load_cnn_batch(img_batch, label_batch,  executor)      
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
        currInput['rgbh'] = img_batch[i]
        #task = partial(_load_batch_helper, currInput)
       # futures.append(executor.submit(task))
        futures.append(executor.submit(_load_batch_helper, currInput))
        #result=_load_batch_helper(currInput)
        #result=executor.map(_load_batch_helper, currInput)
        #results.append(result)

    results = [future.result() for future in futures]
       # results.append( _load_batch_helper(currInput))
    for  i, result in enumerate(results):
        imgdata.append(result[0])
        labels.append(result[1])

    #imgdata = imagenet_utils.preprocess_input(imgdata)
    #imgdata = imgdata / 255.0
    y_train = np.array(labels, np.float32)
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
    rgbh_file = inputDict['rgbh']
    gts_file = inputDict['gts']
    rgbh = cv2.imread(rgbh_file)
 #   rgbh = image.img_to_array(rgbh)
    gts = cv2.imread(gts_file,0)

    y = np.zeros((gts.shape[0], gts.shape[1], 6), dtype=np.float32)
    for i in range(gts.shape[0]):
        for j in range(gts.shape[1]):
            y[i, j, gts[i][j]-1] = 1
#    return y
    #y = image.img_to_array(gts)
    #currOutput = {}
    #currOutput['imgs'] = rgbh
    #currOutput['labels'] = y
  
    return rgbh, y

def GetPredictData(ortho_path, dsm_path,path_size=(256,256),overlap_ratio=0.5,img_ioa=[]):
    img_aug=[]
    from DataAugment import normalize_image_to_path,getAuger_p,getAuger_p_p
    #path_size=(256,256)
   # overlap_ratio=0.5
    [imgs,dsms,image_size]=normalize_image_to_path(ortho_path,dsm_path,path_size,overlap_ratio,work_region=img_ioa,convertLab=False,pad_edge=1,normalize_dsm=1, image_order=1)
    contrast_range_2=(0.9,1.1)
    for i in range(len(imgs)):
        img_data=imgs[i]
        #img_data = img_data.transpose(1,2,0)
        img_data=img_data.astype(np.float32)
        dsm_dat = dsms[i].astype(np.float32)
        dsm_dat = dsm_dat[:,:,np.newaxis]
 #   nDSM_data = nDSM_data[:,:,np.newaxis]
        rgbh=np.concatenate((img_data, dsm_dat), axis=-1)
        for a_id in range (4):
            
            img_a=np.rot90(rgbh,4-a_id)
            img_aug.append(img_a)
        # for a_id in range (4):
        #     auger=getAuger_p_p(a_id)
        #     img_a=auger.augment_image(rgbh)
        #     img_aug.append(img_a)
    return img_aug,image_size

def input_load_train_data(img_train,dsm_train, lable_train,single_id,resize2size=[]):
    inputs=[]
    labels=[]
    from dataFunctions import image_augmentation
    for i in range(len(img_train)):
        rgb_file=img_train[i]
        gts_file=lable_train[i]
        dsm_file=dsm_train[i]
        proj2,geotrans2,img_data=gdal_reader.read_img(rgb_file)
        proj1,geotrans1,label_data=gdal_reader.read_img(gts_file)
        proj,geotrans,DSM_data=gdal_reader.read_img(dsm_file)

    
        img_data=img_data.astype(np.float)
        img_data = img_data.transpose(1,2,0)

        if len(resize2size)>0:
            img_data=cv2.resize(img_data,resize2size)
            label_data=cv2.resize(label_data,resize2size,cv2.INTER_NEAREST)
            DSM_data=cv2.resize(DSM_data,resize2size)
 
        DSM_data = DSM_data.astype(np.float)
        DSM_data = DSM_data[:,:,np.newaxis]
    #   nDSM_data = nDSM_data[:,:,np.newaxis]
        rgbh=np.concatenate((img_data, DSM_data), axis=-1)

        class_label=np.zeros((label_data.shape[0],label_data.shape[1]))
        class_label[(label_data==single_id)]=1
        input_label = to_categorical(class_label, num_classes=2)


        imageMedium,labelMedium = image_augmentation(rgbh, input_label)
        inputs.append(rgbh)
        inputs.append(imageMedium)
        labels.append(input_label)
        labels.append(labelMedium)
    
    inputs = np.array(inputs, np.float32)
    inputs[:,:,:,0:-1]=inputs[:,:,:,0:-1]/125.0-1
    inputs[:,:,:,-1]=inputs[:,:,:,-1]/50.0-1


    return inputs, labels    


from keras import backend as K
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    # dynamic_weights=False
    # if len(weights)<2:
    #     dynamic_weights=True    
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        # if dynamic_weights:
        #     weights=[1,1]
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


def my_weighted_loss(onehot_labels, logits):
    """scale loss based on class weights
    """
    # compute weights based on their frequencies
    beta=tf.reduce_sum(onehot_labels,1);
    beta=tf.reduce_sum(beta,1)
    #beta=tf.count_nonzero(onehot_labels,0)
    beta=K.cast(beta, "float32")/256/256
    class_weights=[[beta[:,1]], [beta[:,0]]];
    class_weights= tf.transpose(class_weights)
    #class_weights =[1,20]  # set your class weights here
    # computer weights based on onehot labels
    class_weights= tf.expand_dims(class_weights, 1);
    #class_weights=tf.expand_dims(class_weights, 1);
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=[onehot_labels], logits=[logits])
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss

def find_all_test_data(data_folder):
    imgs=[]
    dsms=[]
    img_folder=os.path.join(data_folder,'Ortho_result')
    img_files = glob.glob(os.path.join(img_folder, '*.tif'))
    for imgPath in img_files:
        imageName = os.path.split(imgPath)[-1]
        dsmPath=imageName[0:7]+'_DSM.tif'
        imgs.append(imageName)
        dsms.append(dsmPath)



    ###########################
    ortho_list_file=os.path.join(data_folder,'test_orthos.txt')
    dsm_list_file=os.path.join(data_folder,'test_dsm.txt')

    f_ortho = open(ortho_list_file,'w')
    f_dsm = open(dsm_list_file,'w')

    for i in range(len(imgs)):
        f_ortho.write(imgs[i]+'\n');
        f_dsm.write(dsms[i]+'\n');
    f_ortho.close()
    f_dsm.close()    
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
        im=load_img(imgs[0])
        

        vote_map=np.zeros((im.shape[0],im.shape[1],NUM_CATEGORIES))
        for img_p in imgs:
            im=load_img(img_p)
            one_hot=to_categorical(im,NUM_CATEGORIES)
            for i in range(vote_map.shape[-1]):
                vote_map[:,:,i]=vote_map[:,:,i]+one_hot[:,:,i]
        pred=np.argmax(vote_map,axis=-1).astype('uint8')
        if pred.shape[0]>512 or pred.shape[1]>512:
            if m==0:
                offset=[0,0]
            else:
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
    glob_path=os.path.join(result_my,'*CLS.tif')
    files=glob.glob(glob_path)
    baseline_folder=result_folders[1]
#    bridge_folder=result_folders[2]
    for img in files:
        image_name=os.path.split(img)[-1]
        site_name=image_name[0:7]
        result_my=cv2.imread(img,0)
        result_my=convert_labels(result_my,params,toLasStandard=False)
        result_baseline=cv2.imread(os.path.join(baseline_folder,site_name+'_CLS_Merge_all.tif'),0)
        result_baseline=convert_labels(result_baseline,params,toLasStandard=False)

 #       result_bridge=cv2.imread(os.path.join(bridge_folder,image_name),0)
 #       result_bridge=convert_labels(result_bridge,params,toLasStandard=False)

        result_merge=np.zeros([result_baseline.shape[0],result_baseline.shape[1]], dtype = 'uint8')
#           result_merge[(result_baseline==0)]=result_baseline[(result_baseline==0)]
        result_merge[(result_baseline==2)]=result_baseline[(result_baseline==2)]
        #result_merge[(result_my==2)]=result_my[(result_my==2)]
        result_merge[(result_baseline==1)]=result_baseline[(result_baseline==1)]
        result_merge[(result_my==1)]=result_my[(result_my==1)]
        result_merge[(result_baseline==3)]=result_baseline[(result_baseline==3)]
       # result_merge[(result_my==3)]=result_my[(result_my==3)]
        result_merge[(result_baseline==4)]=result_baseline[(result_baseline==4)]
        result_merge[(result_my==4)]=result_my[(result_my==4)]
 #       result_merge[(result_bridge==4)]=result_bridge[(result_bridge==4)]
        result_merge=convert_labels(result_merge,params,toLasStandard=True)
        out_path=os.path.join(out_folder,site_name+'_CLS.tif')
        tifffile.imsave(out_path,result_merge,compress=6)
            
if __name__ == '__main__':
    data_folder='C:/TrainData/Track3/Test-Track3'
    find_all_test_data(data_folder)
 