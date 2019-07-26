import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
import params
import facade_params
from osgeo import gdal, gdalconst
import cv2
import os
import random
from keras.utils.np_utils import to_categorical
import glob
from dataFunctions import convert_labels,load_img
import matplotlib.pyplot as plt
import tifffile
ia.seed(1)
CONTRAST_RANGE=(0.8, 2.0)
PERSPECTIVE_RANGE=(0.05, 0.09)


def getAuger_online(contrast_range,Perspective_range):
    roate=random.randint(0,4)
    seq = iaa.Sequential([
    iaa.Rot90(roate),
    iaa.ContrastNormalization(contrast_range),
    iaa.PerspectiveTransform(scale=Perspective_range),
    ], random_order=False)
    return seq


def augment_online_old(image,label):
    augment_imgs=[]
    augment_label=[]
    augment_imgs.append(image)
    
    currLabel = to_categorical(label, num_classes=5+1)
    currLabel =currLabel[:,:,0:-1]
    augment_label.append(currLabel)
    segmap = ia.SegmentationMapOnImage(label, shape=image.shape, nb_classes=1+5)
    for a_id in range(2):###for each path do 4 augment
        auger=getAuger_online(CONTRAST_RANGE,PERSPECTIVE_RANGE)
        seq_det = auger.to_deterministic() #确定一个数据增强的序列
        images_aug = seq_det.augment_image(image) #将方法应用在原图像上
        segmaps_aug = seq_det.augment_segmentation_maps([segmap]).get_arr_int().astype(np.uint8)
        augment_imgs.append(images_aug)
        currLabel = to_categorical(segmaps_aug, num_classes=5+1)
        currLabel =currLabel[:,:,0:-1]
        augment_label.append(currLabel)
    return augment_imgs,augment_label

def augment_online(image,label):
    """
    the input label is not ont-hot format.
    """
    segmap = ia.SegmentationMapOnImage(label, shape=image.shape, nb_classes=1+5)
    auger=getAuger_online(CONTRAST_RANGE,PERSPECTIVE_RANGE)
    seq_det = auger.to_deterministic() #确定一个数据增强的序列
    images_aug = seq_det.augment_image(image) #将方法应用在原图像上
    segmaps_aug = seq_det.augment_segmentation_maps([segmap]).get_arr_int().astype(np.uint8)
    return images_aug,segmaps_aug

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
    def write_img(self,filename,im_proj,im_geotrans,im_data):
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

        #dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
        #dataset.SetProjection(im_proj)                    #写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])

        del dataset

def normalize_image_to_path(img_path,label_path,path_size,overlap_ratio,work_region=[],resize2size=[],extra_input=False, convertLab=False,pad_edge=1,normalize_dsm=1, image_order=0):
    from grss_data import convertMSI2watertreedata
    if 1:#img_path[:-3]=='tif'):
        #gd_reader = GRID()
        is_msi=0
        if is_msi:
            img=convertMSI2watertreedata(img_path)
        else:
            img=load_img(img_path)
            #aa=0
        #img=img.transpose(2,0,1)
        image_order=0
        #proj,geotrans,img = gd_reader.read_img(img_path)
        if len(label_path)>3:
            extra_input=True
            label=load_img(label_path)
            #proj,geotrans,label = gd_reader.read_img(label_path)
        else:
            label=img
            #label_input=False
    else:
        img=img_path
        label=label_path


    if extra_input:
        if convertLab:
            label=convertLas2Train(label, params.LABEL_MAPPING_LAS2TRAIN)
        elif normalize_dsm:
            nan_data=np.isnan(label)
            label[nan_data] = 99999
            min_t=label.min()
            label=label-min_t
            label[nan_data]=0

    if len(work_region)>0:
        img = img[:,work_region[0]:work_region[1],work_region[2]:work_region[3]]
        if extra_input:
            if (label.shape[0]>work_region[1]-work_region[0] or label.shape[1]>work_region[3]-work_region[2]):
                label=label[:,work_region[0]:work_region[1],work_region[2]:work_region[3]]

    if len(resize2size)>0:
        if image_order==1:
            img = img.transpose(1,2,0)
        img=cv2.resize(img,resize2size)
        if image_order==1:
            img=img.transpose(2,0,1)
        if extra_input:
            label=cv2.resize(label,resize2size,interpolation=cv2.INTER_NEAREST)
    



    imgs=[]
    labels=[]


    if img.shape[0]==path_size[0] and img.shape[1]==path_size[1]:
        imgs.append(img)
        labels.append(label)
        return imgs,labels,[path_size[0],path_size[1]]
    else:
        if img.shape[0]<path_size[0]:
            padded_img= np.zeros((path_size[0],img.shape[1],img.shape[2]), dtype=img.dtype)
            padded_label= np.ones((path_size[0],img.shape[1],img.shape[2]), dtype=label.dtype)*facade_params.train_background
            padding_step=round((path_size[0]-img.shape[0])/2)
            padded_img[padding_step:padding_step+img.shape[0],:,:]=img
            padded_label[padding_step:padding_step+img.shape[0],:,:]=label
            img=padded_img
            label=padded_label
        if img.shape[1]<path_size[1]:
            padded_img= np.zeros((img.shape[0],path_size[1],img.shape[2]), dtype=img.dtype)
            padded_label= np.ones((img.shape[0],path_size[1],img.shape[2]), dtype=label.dtype)*facade_params.train_background
            padding_step=round((path_size[1]-img.shape[1])/2)
            padded_img[:,padding_step:padding_step+img.shape[1],:]=img
            padded_label[:,padding_step:padding_step+img.shape[1],:]=label
            img=padded_img
            label=padded_label

    # cv2.imshow('input_img',img)
    # cv2.imshow('label',label*20)
    # cv2.waitKeyEx()
    if image_order==0:
        rows=img.shape[0]
        cols=img.shape[1]
    elif image_order==1:
        rows=img.shape[1]
        cols=img.shape[2]    
    if(1):
    #        patch_ranges=calculate_cut_range(img.shape[1:3], patch_size=[path_size[0],path_size[1]],overlap=overlap_ratio)


        patch_height = path_size[0]
        patch_width = path_size[1]
        width_overlap = patch_width * overlap_ratio
        height_overlap = patch_height *overlap_ratio
        x_e = 0
        while (x_e < cols):
            y_e=0
            x_s = max(0, x_e - width_overlap);
            x_e = x_s + patch_width;
            if (x_e > cols):
                x_e = cols
            if (pad_edge == 1): ## if the last path is not enough, then extent to the inerside.
                x_s = x_e - patch_width
            if (pad_edge == 2):## if the last patch is not enough, then extent to the outside(with black).
                x_s=x_s
            while (y_e < rows):
                y_s = max(0, y_e - height_overlap);
                y_e = y_s + patch_height;
                if (y_e > rows):
                    y_e = rows;
                if (pad_edge == 1): ## if the last path is not enough, then extent to the inerside.
                    y_s = y_e - patch_height
                if (pad_edge == 2):## if the last patch is not enough, then extent to the outside(with black).
                    y_s=y_s
            #    range_1=[int(y_s),int(y_e)]
                if image_order==0:
                    img_patch=img[int(y_s):int(y_e),int(x_s):int(x_e),:]
                else:
                    img_patch=img[:,int(y_s):int(y_e),int(x_s):int(x_e)]
                if extra_input:
                    label_patch=label[int(y_s):int(y_e),int(x_s):int(x_e)]
                imgs.append(img_patch)
                if extra_input:
                    labels.append(label_patch)
    return imgs,labels,[rows,cols]

def img2patches(img, patch_size,overlap_rati):
    patches=[]

    patch_range=calculate_cut_range(img.shape[0:2],patch_size,overlap_rati)
    for id in range(len(patch_range)):
        y_s=round(patch_range[id][0])
        y_e=round(patch_range[id][1])
        x_s=round(patch_range[id][2])
        x_e=round(patch_range[id][3])
        patch=img[y_s:y_e,x_s:x_e,:]
        patches.append(patch)
    return patches

def calculate_cut_range(img_size, patch_size,overlap,pad_edge=1):
    patch_range=[]
    patch_height = patch_size[0]
    patch_width = patch_size[1]
    width_overlap = patch_width * overlap
    height_overlap = patch_height *overlap
    cols=img_size[1]
    rows=img_size[0]
    x_e = 0
    while (x_e < cols):
        y_e=0
        x_s = max(0, x_e - width_overlap);
        x_e = x_s + patch_width;
        if (x_e > cols):
            x_e = cols
        if (pad_edge == 1): ## if the last path is not enough, then extent to the inerside.
            x_s = x_e - patch_width
        if (pad_edge == 2):## if the last patch is not enough, then extent to the outside(with black).
            x_s=x_s
        while (y_e < rows):
            y_s = max(0, y_e - height_overlap);
            y_e = y_s + patch_height;
            if (y_e > rows):
                y_e = rows;
            if (pad_edge == 1): ## if the last path is not enough, then extent to the inerside.
                y_s = y_e - patch_height
            if (pad_edge == 2):## if the last patch is not enough, then extent to the outside(with black).
                y_s=y_s
            patch_range.append([int(y_s),int(y_e),int(x_s),int(x_e)])
    return patch_range
def crop_normalized_path(img_folder, label_folder,out_folder,path_size,overlap_ratio):
    if os.path.exists(out_folder)==0:
        os.makedirs(out_folder)
    sub_img_folder=os.path.join(out_folder,'img_patch')
    if os.path.exists(sub_img_folder)==0:
        os.makedirs(sub_img_folder)
    sub_label_folder=os.path.join(out_folder,'label_patch')
    if os.path.exists(sub_label_folder)==0:
        os.makedirs(sub_label_folder)
    img_list_file=os.path.join(out_folder,'img_list.txt')
    label_list_file=os.path.join(out_folder,'label_list.txt')
    f_img = open(img_list_file,'w')
    f_label = open(label_list_file,'w')
    for filename in os.listdir(img_folder):    ##JAX_004_007_CLS  JAX_004_007_RGB
        file_apx=filename[-3:]
        if file_apx=='jpg' or  file_apx=='png' or  file_apx=='tif':
            label_path=os.path.join(label_folder , filename[:-7]+'CLS.tif')
            img_path=os.path.join(img_folder,filename)
            if os.path.exists(label_path):
                [imgs,labels]=normalize_image_to_path(img_path,label_path,path_size,overlap_ratio,convertLab=True,pad_edge=1)
                for i in range(len(imgs)):
                    img_write_path=sub_img_folder+'/'+filename[:-4]+'_'+str(i)+'.tif'
                    label_write_path=sub_label_folder+'/'+filename[:-4]+'_'+str(i)+'.tif'
                    cv2.imwrite(img_write_path,imgs[i])
                    #sio.savemat(label_write_path,mdict={'label_map':labels[i]})
                    cv2.imwrite(label_write_path,labels[i])
                    f_img.write('img_patch/'+filename[:-4]+'_'+str(i)+'.tif'+'/n');
                    f_label.write('label_patch/'+filename[:-4]+'_'+str(i)+'.tif'+'/n');
    f_img.close()
    f_label.close()

def convertLas2Train(Lorig,labelMapping):
    L = Lorig.copy()
    for key,val in labelMapping.items():
        L[Lorig==key] = val
    return L
def getAuger(rot_angle,contrast_range,Perspective_range):
    if rot_angle==0:
        seq = iaa.Sequential([
        iaa.ContrastNormalization(contrast_range),
        iaa.PerspectiveTransform(scale=Perspective_range),
        ], random_order=False)
    else:
        seq = iaa.Sequential([
    #iaa.Fliplr(0.5),
        iaa.Rot90(rot_angle),
        iaa.ContrastNormalization(contrast_range),
        iaa.PerspectiveTransform(scale=Perspective_range),
        ], random_order=False)
    return seq

def getAuger_p(rot_angle,contrast_range):
    if rot_angle==0:
        seq = iaa.Sequential([
        iaa.ContrastNormalization(contrast_range),
        ], random_order=False)
    else:
        seq = iaa.Sequential([
    #iaa.Fliplr(0.5),
        iaa.Rot90(rot_angle),
        iaa.ContrastNormalization(contrast_range),
        ], random_order=False)
    return seq
import math    


def getAuger_p_p(rot_angle):

    seq = iaa.Sequential([
    #iaa.Fliplr(0.5),
    iaa.Rot90(rot_angle),
    ], random_order=False)
    return seq   
def goback_auger_p(imgs):
    img_back=[]
    #ia.seed(1)
    contrast_range=(0.0,0.0)
    for id in range(len(imgs)):
        img=imgs[id]
        i=id%4
        if i==0:
            img_back.append(img)
        else:
            for mmm in range (i):
                img=np.squeeze(img)
                img=np.rot90(img)
                img = np.expand_dims(img, axis=0)
#            auger=getAuger_p_p(4-i)
#            img_a=auger.augment_image(img)
            img_back.append(img)
    return img_back

def GetPatchWeight(patch_size=[256,256],pad=32,last_value=0.4):
    patch_wid=patch_size[1]
    patch_hei=patch_size[0]
    patch_weights=np.zeros((patch_hei,patch_wid))
    # for the u-net the resize is 32 times, hence, the edge pad need to be 16 pixels. 
   #pad=32

    x_good_d=patch_wid/2-pad
    y_good_d=patch_hei/2-pad
    center=[patch_hei/2.,patch_wid/2.]
    for y in range (patch_hei):
        for x in range(patch_wid):
            dis_y=abs(center[0]-y)
            dis_x=abs(center[1]-x)
            if dis_x>x_good_d or dis_y>y_good_d:
                dis=max(dis_x-x_good_d,dis_y-y_good_d)
                weight=1-dis/pad*(1-last_value)
            else:
                weight=1
            patch_weights[y,x]=weight
    return patch_weights
from track3_data import get_batch_inds
def patch2img_height(patches,img_size,patch_weights,num_class=5, overlap=0.5):
    patches=np.squeeze(patches)
    patch_wid=patches[0].shape[1]
    patch_hei=patches[0].shape[0]
    vote=np.zeros((img_size[0],img_size[1]))
    numbers=np.zeros((img_size[0],img_size[1]))
    patch_ranges=calculate_cut_range(img_size, patch_size=[patch_hei,patch_wid],overlap=overlap)
    path_vote=np.ones((patch_hei,patch_wid))
    N=len(patches)
    idx = range(N)
   # patch_weights = patch_weights[:,:,np.newaxis]
    batchInds = get_batch_inds(4, idx, N) 
    for inds in range(len(batchInds)):
        img_batchs = patches[ batchInds[inds]]
 #        img_batch=np.array(img_batchs,np.float32)
        for id in range(len(img_batchs)):
            patch=img_batchs[id]
            patch=np.squeeze(patch)
            patch=np.array(patch,np.float32)
            #patch_mean=np.mean(patch)
            
           # currLabel = to_categorical(patch, num_classes=num_class)

 #           img_=reduce(img_batchs,0)
            y_s=round(patch_ranges[inds][0])
            y_e=round(patch_ranges[inds][1])
            x_s=round(patch_ranges[inds][2])
            x_e=round(patch_ranges[inds][3])
            weighted_patch=patch*patch_weights
            #for i in range(5):
            vote[y_s:y_e,x_s:x_e]=vote[y_s:y_e,x_s:x_e]+weighted_patch
            numbers[y_s:y_e,x_s:x_e]=numbers[y_s:y_e,x_s:x_e]+patch_weights

    # for i in range(len(patches)):
    #     patch_id=i//4
    #     patch=np.squeeze(patches[i])
    #     for x in range(patch_wid):
    #         for y in range(patch_hei):
    #             id_y=round(patch_ranges[patch_id][0])+y
    #             id_x=round(patch_ranges[patch_id][2])+x
    #             c=patch[y,x]
    #             vote[id_y,id_x,c]=vote[id_y,id_x,c]+1#patch_weights[y,x]
    #pred = np.argmax(vote, axis=-1).astype('uint8')
    pred = vote/numbers
    pred=np.array(pred,np.float32)
    return pred

def Patch2Img(patches,img_size,patch_weights,num_class=5, overlap=0.5):
    patches=np.squeeze(patches)
    patch_wid=patches[0].shape[1]
    patch_hei=patches[0].shape[0]
    vote=np.zeros((img_size[0],img_size[1],num_class))
    patch_weights = patch_weights[:,:,np.newaxis]
    patch_ranges=calculate_cut_range(img_size, patch_size=[patch_hei,patch_wid],overlap=overlap)
    for id in range(len(patches)):
            patch=patches[id]
            currLabel = to_categorical(patch, num_classes=num_class) #####
           # currLabel=patch  ##track2
            y_s=round(patch_ranges[id][0])
            y_e=round(patch_ranges[id][1])
            x_s=round(patch_ranges[id][2])
            x_e=round(patch_ranges[id][3])
            weighted_patch=currLabel*patch_weights
            #for i in range(5):
            vote[y_s:y_e,x_s:x_e,:]=vote[y_s:y_e,x_s:x_e,:]+weighted_patch
    pred = np.argmax(vote, axis=-1).astype('uint8')
    #pred=vote
    return pred


def Patch2Img(patches,img_size,patch_weights,num_class=5, overlap=0.5):
    patches=np.squeeze(patches)
    if len(patches.shape)<3:
        patch_wid=patches.shape[1]
        patch_hei=patches.shape[0]        
    else:
        patch_wid=patches[0].shape[1]
        patch_hei=patches[0].shape[0]
    vote=np.zeros((img_size[0],img_size[1],num_class))
    if patch_weights.shape[0]!=patch_hei or patch_weights.shape[1]!=patch_wid:
        new_patches=[]
        for i in range(len(patches)):
            new_patches.append(cv2.resize(patches[i],(patch_weights.shape[1],patch_weights.shape[0]),cv2.INTER_NEAREST))
        patches=new_patches
    patch_ranges=calculate_cut_range(img_size, patch_size=[patch_hei,patch_wid],overlap=overlap)
    if len(patches.shape)<3:
        N=1
    else:
        N=len(patches)
    idx = range(N)
    patch_weights = patch_weights[:,:,np.newaxis]
    batchInds = get_batch_inds(4, idx, N) 
    for inds in range(len(batchInds)):
        img_batchs = patches[ batchInds[inds]]
 #        img_batch=np.array(img_batchs,np.float32)
        for id in range(len(img_batchs)):
            patch=img_batchs[id]
            patch=np.squeeze(patch)
            currLabel = to_categorical(patch, num_classes=num_class)

 #           img_=reduce(img_batchs,0)
            y_s=round(patch_ranges[inds][0])
            y_e=round(patch_ranges[inds][1])
            x_s=round(patch_ranges[inds][2])
            x_e=round(patch_ranges[inds][3])
            weighted_patch=currLabel*patch_weights
            #for i in range(5):
            vote[y_s:y_e,x_s:x_e,:]=vote[y_s:y_e,x_s:x_e,:]+weighted_patch

    # for i in range(len(patches)):
    #     patch_id=i//4
    #     patch=np.squeeze(patches[i])
    #     for x in range(patch_wid):
    #         for y in range(patch_hei):
    #             id_y=round(patch_ranges[patch_id][0])+y
    #             id_x=round(patch_ranges[patch_id][2])+x
    #             c=patch[y,x]
    #             vote[id_y,id_x,c]=vote[id_y,id_x,c]+1#patch_weights[y,x]
    pred = np.argmax(vote, axis=-1).astype('uint8')
    return pred

def predict_data_generator(img_path):
    img_aug=[]
    ia.seed(1)
    contrast_range=(0.8, 2.0)
    contrast_range_2=(0.9,1.1)
    path_size=(512,512)
    overlap_ratio=0.5
    label_path=img_path
    if os.path.exists(label_path):
        [imgs,labels]=normalize_image_to_path(img_path,label_path,path_size,overlap_ratio,convertLab=True,pad_edge=1)
        for img_patch in imgs:
            for a_id in range (4):
                auger=getAuger_p(a_id,contrast_range)
                img_a=auger.augment_image(img_patch)
                img_aug.append(img_a)
            for a_id in range (4):
                auger=getAuger_p(a_id,contrast_range_2)
                img_a=auger.augment_image(img_patch)
                img_aug.append(img_a)
    return img_aug;
                        

def imageAugment(datafolder):

    img_files=[]
    label_files=[]

####find images
    file=os.path.join(datafolder,'img_list.txt')
    fp = open(file)
    lines = fp.readlines()
    fp.close()
    num_img = len(lines)
    for line in lines:
            line = line.strip('/n')           
            img_files.append(line)
####find labels
    file=os.path.join(datafolder,'label_list.txt')
    fp = open(file)
    lines = fp.readlines()
    fp.close()
    for line in lines:
            line = line.strip('/n')           
            label_files.append(line)
    num_label=len(lines)

    if num_img !=num_label:
          return       
####set augment parameters
    ia.seed(1)
    contrast_range=(0.8, 2.0)
    Perspective_range=(0.05, 0.09)

    img_list_file=os.path.join(datafolder,'img_list_au.txt')
    label_list_file=os.path.join(datafolder,'label_list_au.txt')
    f_img = open(img_list_file,'w')
    f_label = open(label_list_file,'w')

    for idx  in range (num_img):
        label=cv2.imread(datafolder+label_files[idx],0)
        image=cv2.imread(datafolder+img_files[idx])
       # label= convertLas2Train(label, params.LABEL_MAPPING_LAS2TRAIN)
        segmap = ia.SegmentationMapOnImage(label, shape=image.shape, nb_classes=1+5)
        for a_id in range(4):###for each path do 4 augment
            auger=getAuger(a_id,contrast_range,Perspective_range)
            seq_det = auger.to_deterministic() #确定一个数据增强的序列
            images_aug = seq_det.augment_image(image) #将方法应用在原图像上
            segmaps_aug = seq_det.augment_segmentation_maps([segmap])[0].get_arr_int().astype(np.uint8)
            img_write_path=img_files[idx][:-4]+'_{}.tif'.format(a_id)
            label_write_path=label_files[idx][:-4]+'_{}.tif'.format(a_id)
            cv2.imwrite(datafolder+img_write_path,images_aug)
            cv2.imwrite(datafolder+label_write_path,segmaps_aug)
            f_img.write('img_patch/'+img_write_path+'/n');
            f_label.write('label_patch/'+label_write_path+'/n');

    f_img.close()
    f_label.close()

def dataAugument(img_folder, label_folder, out_folder,path_size,overlap_ratio):
#    crop_normalized_path(img_folder,label_folder,out_folder,path_size,overlap_ratio,)
    imageAugment(out_folder)
def crop_normalized_patch_track3(img_folder, label_folder,out_folder,path_size,overlap_ratio):
    if os.path.exists(out_folder)==0:
        os.makedirs(out_folder)
    sub_img_folder=os.path.join(out_folder,'img_patch')
    if os.path.exists(sub_img_folder)==0:
        os.makedirs(sub_img_folder)
    sub_label_folder=os.path.join(out_folder,'label_patch')
    if os.path.exists(sub_label_folder)==0:
        os.makedirs(sub_label_folder)
    sub_dsm_folder=os.path.join(out_folder,'dsm_patch')
    if os.path.exists(sub_dsm_folder)==0:
        os.makedirs(sub_dsm_folder)

    img_list_file=os.path.join(out_folder,'img_list.txt')
    label_list_file=os.path.join(out_folder,'label_list.txt')
    dsm_list_file=os.path.join(out_folder,'dsm_list.txt')
    f_img = open(img_list_file,'w')
    f_label = open(label_list_file,'w')
    f_dsm = open(dsm_list_file,'w')
    gd_reader = GRID()
    for filename in os.listdir(img_folder):    ##JAX_004_007_CLS  JAX_004_007_RGB
        file_apx=filename[-3:]
        if file_apx=='jpg' or  file_apx=='png' or  file_apx=='tif':
            label_path=os.path.join(label_folder , filename[:-17]+'CLS.tif')
            dsm_path=os.path.join(label_folder , filename[:-17]+'DSM.tif')
            img_path=os.path.join(img_folder,filename)
            if os.path.exists(label_path):
                [imgs,labels]=normalize_image_to_path(img_path,label_path,path_size,overlap_ratio,convertLab=True,pad_edge=1)
                [imgs,dsms]=normalize_image_to_path(img_path,dsm_path,path_size,overlap_ratio,convertLab=False,pad_edge=1)
                for i in range(len(imgs)):
                    img_write_path=sub_img_folder+'/'+filename[:-4]+'_'+str(i)+'.tif'
                    label_write_path=sub_label_folder+'/'+filename[:-10]+'_CLS_'+str(i)+'.tif'
                    dsm_write_path=sub_dsm_folder+'/'+filename[:-10]+'_DSM_'+str(i)+'.tif'
                    gd_reader.write_img(img_write_path,[],[],imgs[i]) #写数据
                    gd_reader.write_img(dsm_write_path,[],[],dsms[i]) #写数据
                    gd_reader.write_img(label_write_path,[],[],labels[i]) #写数据

                    f_img.write('img_patch/'+filename[:-4]+'_'+str(i)+'.tif'+'/n');
                    f_label.write('label_patch/'+filename[:-10]+'_CLS_'+str(i)+'.tif'+'/n');
                    f_dsm.write('dsm_patch/'+filename[:-10]+'_DSM_'+str(i)+'.tif'+'/n');    
    f_img.close()
    f_label.close()
def crop_normalized_patch_track1(img_folder, label_folder,out_folder,path_size,overlap_ratio):
    resize2size=()
    if os.path.exists(out_folder)==0:
        os.makedirs(out_folder)
    sub_img_folder=os.path.join(out_folder,'img_patch')
    if os.path.exists(sub_img_folder)==0:
        os.makedirs(sub_img_folder)
    sub_label_folder=os.path.join(out_folder,'label_patch')#
    if os.path.exists(sub_label_folder)==0:
        os.makedirs(sub_label_folder)
    img_list_file=os.path.join(out_folder,'img_list.txt')
    label_list_file=os.path.join(out_folder,'label_list.txt')#
    f_img = open(img_list_file,'w')
    f_label = open(label_list_file,'w')
    gd_reader = GRID()
    for filename in os.listdir(img_folder):    ##JAX_004_007_CLS  JAX_004_007_RGB
        file_apx=filename[-3:]
        if file_apx=='jpg' or  file_apx=='tif': ##or  file_apx=='png' 
            ##label_path=os.path.join(label_folder , filename[:-7]+'AGL.tif')
            label_path=os.path.join(label_folder , filename[:-3]+'png')
            img_path=os.path.join(img_folder,filename)
            if os.path.exists(label_path):
                [imgs,labels,size_o]=normalize_image_to_path(img_path,label_path,path_size,overlap_ratio,resize2size=resize2size,convertLab=False,pad_edge=1,normalize_dsm=0,image_order=1)
                for i in range(len(imgs)):
                   # img_write_path=sub_img_folder+'/'+filename[:-4]+'_'+str(i)+'.tif'
                    #label_write_path=sub_label_folder+'/'+filename[:-8]+'_AGL_'+str(i)+'.tif'#
                    #gd_reader.write_img(img_write_path,[],[],imgs[i]) #写数据
                    #gd_reader.write_img(label_write_path,[],[],labels[i]) #写数据

                    img_write_path=sub_img_folder+'/'+filename[:-4]+'_'+str(i)+'.jpg'
                    label_write_path=sub_label_folder+'/'+filename[:-4]+'_'+str(i)+'.png'#
                    cv2.imwrite(img_write_path,imgs[i])
                    cv2.imwrite(label_write_path,labels[i])                    
                    f_img.write(filename[:-4]+'_'+str(i)+'.jpg'+'\n');
                    f_label.write(filename[:-4]+'_'+str(i)+'.png'+'\n');#
    f_img.close()
    f_label.close()

def AnalyzSampleCategory(label_folder):

    lists_1=[]
    lists_2=[]
    for i in range(facade_params.Num_Category):
        lists_1.append([])
        lists_2.append([])
    ratios=[0.02]*facade_params.Num_Category#'What is this? 1: window; 2: wall; 3: door; 4: balcony; 5: others;0:background '
    count=0
    for filename in os.listdir(label_folder):    ##JAX_004_007_CLS  JAX_004_007_RGB
        file_apx=filename[-3:]
        if file_apx=='png' or  file_apx=='tif':
            label=cv2.imread(os.path.join(label_folder,filename),0)
            num_class=label.max()
           # label=convertLas2Train(label, params.LABEL_MAPPING_LAS2TRAIN)
            img_size=label.shape[0]*label.shape[1]
            count=count+1
            for i in range(num_class):
                mask=(label==i)
                y_new = label[mask]
                aa=y_new.size/img_size
                # if aa>ratios[i]:
                #     lists_1[i].append(filename)
                # else:
                #     lists_2[i].append(filename)
                if aa>ratios[i]:
                    lists_1[i].append(str(count))
                else:
                    lists_2[i].append(str(count))
    
    return lists_1,lists_2



def AnalyzCategoryHeight(label_folder):
    class_names=['ground','tree','roof','water','bridge']
    lists_1=[210*0]*params.NUM_CATEGORIES

    img_files = glob.glob(os.path.join(label_folder, '*CLS.tif'))
    for filename in img_files:    ##JAX_004_007_CLS  JAX_004_007_RGB

        label=load_img(os.path.join(label_folder,filename))
        dsm=load_img(os.path.join(label_folder,filename.replace('CLS','AGL')))
        label=convertLas2Train(label, params.LABEL_MAPPING_LAS2TRAIN)
        max_dsm=np.max(dsm)
        if max_dsm>70:
            print('max is:',max_dsm)
            print('in image:',filename)
        min_dsm=np.min(dsm)
        if min_dsm<-3:
            print('min is:',min_dsm)
            print('in image:',filename)
        find_nan = np.where(dsm == dsm, 1, 0)
        if np.min(find_nan)==0:
            print('NAN in image:',filename)
        for i in range(params.NUM_CATEGORIES):
            mask=(label==i)
            cc=dsm[mask]
            hist, _ = np.histogram(cc, range=(-1, 200), bins=201)
            hist_av=hist/dsm.shape[1]/dsm.shape[0]
            lists_1[i]=lists_1[i]+hist_av
            #############
    for i in range(params.NUM_CATEGORIES):
        data=lists_1[i]
        plt.plot(range(-1,200),data)
        #plt.hist(data,201,range=(-1,200))
    #设置出横坐标
        plt.xlabel('height(m)')
    #设置纵坐标的标题
        plt.ylabel('Number of occurences')
    #设置整个图片的标题
        plt.title('Frequency distribution of '+class_names[i] )

    # 展示出我们的图片
        plt.show()

    return lists_1

def GenerateMSIdata(img_folder,label_folder,out_folder,path_size,overlap_ratio):
    resize2size=()
    if os.path.exists(out_folder)==0:
        os.makedirs(out_folder)
    sub_img_folder=os.path.join(out_folder,'msi_patch')
    if os.path.exists(sub_img_folder)==0:
        os.makedirs(sub_img_folder)
    sub_label_folder=os.path.join(out_folder,'label_patch')#
    if os.path.exists(sub_label_folder)==0:
        os.makedirs(sub_label_folder)
    img_list_file=os.path.join(out_folder,'msi_list.txt')
    label_list_file=os.path.join(out_folder,'label_list.txt')#
    f_img = open(img_list_file,'w')
    f_label = open(label_list_file,'w')
    gd_reader = GRID()
    for filename in os.listdir(img_folder):    ##JAX_004_007_CLS  JAX_004_007_RGB
        file_apx=filename[-3:]
        if file_apx=='jpg' or  file_apx=='png' or  file_apx=='tif':
            label_path=os.path.join(label_folder , filename[:-7]+'CLS.tif')
            img_path=os.path.join(img_folder,filename)
            if 1:#os.path.exists(label_path):
                [imgs,labels,size_o]=normalize_image_to_path(img_path,label_path,path_size,overlap_ratio,label_input=False,resize2size=resize2size,convertLab=False,pad_edge=1,normalize_dsm=0)
                for i in range(len(imgs)):
                    img_write_path=sub_img_folder+'/'+filename[:-4]+'_'+str(i)+'.tif'
                    #label_write_path=sub_label_folder+'/'+filename[:-8]+'_AGL_'+str(i)+'.tif'#
                    gd_reader.write_img(img_write_path,[],[],imgs[i]) #写数据
                    #gd_reader.write_img(label_write_path,[],[],labels[i]) #写数据
                    f_img.write(filename[:-4]+'_'+str(i)+'.tif'+'/n');
                    f_label.write(filename[:-8]+'_CLS_'+str(i)+'.tif'+'/n');#
    f_img.close()
    f_label.close()
def HardSample(img_folder, label_folder, prediction_folder):
    from track1_metrics import compute_metrics
    check_folder='G:/programs/dfc2019-master/data/checkfolder2'
    if os.path.exists(check_folder)==0:
        os.makedirs(check_folder)
    glob_path=os.path.join(label_folder,'*.tif')
    files=glob.glob(glob_path)
    bad_ids=[]
    erres=[]
    class_nums=[]
    cc=1
    hard_list_file=os.path.join(check_folder,'hard_list.txt')
    error_file=os.path.join(check_folder,'error.txt')
    number_file=os.path.join(check_folder,'number.txt')
    f_hard = open(hard_list_file,'w')
    f_erre = open(error_file,'w')
    f_num = open(number_file,'w')

    for file in files:
        img_name=os.path.split(file)[-1]
        
        predict=os.path.join(prediction_folder,img_name)
        #predict=load_img(os.path.join(prediction_folder,img_name))
        error,nums=compute_metrics(predict,file)
        #true_error=error[error>0]
        true_error=error[nums>8242]
        if np.min(true_error)<0.5:
            bad_ids.append(cc)
            erres.append(error)
            class_nums.append(nums)
            img=load_img(predict)*50
            label=load_img(file)*50
            immm=load_img(os.path.join(img_folder,img_name.replace('CLS','RGB')))
            tifffile.imsave(os.path.join(check_folder,img_name), img, compress=6)
            tifffile.imsave(os.path.join(check_folder,img_name[:-4]+'_.tif'), label, compress=6)
            tifffile.imsave(os.path.join(check_folder,img_name[:-4]+'_s.tif'), immm, compress=6)
            f_hard.write(str(cc))
            f_hard.write('\n')
            
            f_erre.write(str(list(error)))
            f_erre.write('\n')
            f_num.write(str(list(nums)))
            f_num.write('\n')
        cc=cc+1
    f_hard.close()
    f_erre.close()
    f_num.close()

def HardHeightSamples(img_folder, label_folder, dsm_folder, prediction_folder):
    from track1_metrics import compute_height
    check_folder='G:/programs/dfc2019-master/data/height_checkfolder_bridge'
    if os.path.exists(check_folder)==0:
        os.makedirs(check_folder)
    glob_path=os.path.join(dsm_folder,'*AGL.tif')
    files=glob.glob(glob_path)
    bad_ids=[]
    erres=[]
    class_nums=[]
    cc=1
    hard_list_file=os.path.join(check_folder,'hard_list.txt')
    error_file=os.path.join(check_folder,'error.txt')
    number_file=os.path.join(check_folder,'number.txt')
    f_hard = open(hard_list_file,'w')
    f_erre = open(error_file,'w')
    f_num = open(number_file,'w')

    for file in files:
        img_name=os.path.split(file)[-1]
        
        predict=os.path.join(prediction_folder,img_name)
        #predict=load_img(os.path.join(prediction_folder,img_name))
        label=os.path.join(label_folder,img_name.replace('AGL','CLS'))
        accurary,nums=compute_height(predict,file,label)
        if accurary[4]<0.5 and accurary[4]>0.01 and nums[4]>4000:
            stop_here=1
        # accurary=accurary[nums>8242]
        # if np.min(accurary)<0.5:
            bad_ids.append(cc)
            erres.append(accurary)
            class_nums.append(nums)
            pre=load_img(predict)#*50
            tru_dsm=load_img(file)#*50
            class_label=load_img(label)
            immm=load_img(os.path.join(img_folder,img_name.replace('AGL','RGB')))
            tifffile.imsave(os.path.join(check_folder,img_name), pre, compress=6)
            tifffile.imsave(os.path.join(check_folder,img_name[:-4]+'_.tif'), tru_dsm, compress=6)
            tifffile.imsave(os.path.join(check_folder,img_name[:-4]+'_s.tif'), immm, compress=6)
            tifffile.imsave(os.path.join(check_folder,img_name[:-4]+'_label.tif'), class_label, compress=6)
            f_hard.write(str(cc))
            f_hard.write('\n')
            
            f_erre.write(str(list(accurary)))
            f_erre.write('\n')
            f_num.write(str(list(nums)))
            f_num.write('\n')
        cc=cc+1
    f_hard.close()
    f_erre.close()
    f_num.close()
   
def HeightScale(img_folder, label_folder, dsm_folder, prediction_folder):
    from track1_metrics import ComputScale
    
    import matplotlib.pyplot as plt

    check_folder='G:/programs/dfc2019-master/data/height_scales'
    if os.path.exists(check_folder)==0:
        os.makedirs(check_folder)
    glob_path=os.path.join(dsm_folder,'*AGL.tif')
    files=glob.glob(glob_path)
    true_values=[]
    pre_values=[]
    ################
    for file in files:
        img_name=os.path.split(file)[-1]
        
        predict=os.path.join(prediction_folder,img_name)
        #predict=load_img(os.path.join(prediction_folder,img_name))
        label=os.path.join(label_folder,img_name.replace('AGL','CLS'))
        truth_height,pre_height=ComputScale(predict,file,label,lab_num=1)
        true_values.extend(truth_height)
        pre_values.extend(pre_height)
    np.save("true_values_tree_2.npy",true_values)
    np.save("pre_values_tree_2.npy",pre_values)
    ###############
    true_values= np.load("true_values_tree_2.npy")
    pre_values= np.load("pre_values_tree_2.npy")+10
    #用3次多项式拟合
    x=pre_values
    y=true_values
    bef_eror=np.sum(abs(np.array(x)-y))/len(x)
    f1 = np.polyfit(x, y, 2)
   # res=f1.
    p1 = np.poly1d(f1)
    print(p1)
    
    #也可使用yvals=np.polyval(f1, x)
    yvals = p1(x)  #拟合y值
    eror=np.sum(abs(yvals-y))/len(yvals)
    bef_eror=np.sum(abs(np.array(x)-y))/len(x)
    print('error: ', eror)
    print(', bef_eror: ', bef_eror)
    #绘图
    plot1 = plt.plot(x[0:5000], y[0:5000], 's',label='original values')
    plot2 = plt.plot(x[0:5000], yvals[0:5000], 'r',label='polyfit values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4) #指定legend的位置右下角
    plt.title('polyfitting')
    plt.show()


def DataSampleAnalysis(img_folder,label_folder,out_folder,path_size,overlap_ratio):
    driver = gdal.GetDriverByName('HFA')
    driver.Register()
    #crop_normalized_patch_track1(img_folder,label_folder,out_folder,path_size,overlap_ratio)
 
    patch_lable=os.path.join(out_folder,'label_patch/')
    lists_1,lists_2=AnalyzSampleCategory(patch_lable)##'What is this? 1: window; 2: wall; 3: door; 4: balcony; 5: others;0:background '
    num_samples=len(lists_1[0])+len(lists_2[0])
    ratios=[]
    txt_files=[]
    txt_folder=os.path.join(out_folder,'class_file_record')
    if not os.path.exists(txt_folder):
        os.mkdir(txt_folder)

    for i in range(len(lists_1)):
        ratios.append(len(lists_1[i])/num_samples)
        txt_files.append(facade_params.category_names[i])

    for i in range(len(txt_files)):
        file_path=os.path.join(txt_folder,txt_files[i]+'.txt')
        f=open(file_path,'w')
        for img in lists_1[i]:
            f.write(img+'\n');
        f.close()


    # # ground_list_file=os.path.join(out_folder,'ground_list.txt')
    # # tree_list_file=os.path.join(out_folder,'tree_list.txt')
    # # roof_list_file=os.path.join(out_folder,'roof_list.txt')
    # # water_list_file=os.path.join(out_folder,'water_list.txt')
    # # bridge_list_file=os.path.join(out_folder,'bridge_list.txt')
    # background_list_file=os.path.join(out_folder,'background.txt')
    # ground_list_file=os.path.join(out_folder,'window.txt')
    # tree_list_file=os.path.join(out_folder,'wall.txt')
    # roof_list_file=os.path.join(out_folder,'door.txt')
    # water_list_file=os.path.join(out_folder,'balcony.txt')
    # bridge_list_file=os.path.join(out_folder,'orths.txt')

    # f_background = open(background_list_file,'w')

    # f_ground = open(ground_list_file,'w')
    # f_tree = open(tree_list_file,'w')
    # f_roof = open(roof_list_file,'w')
    # f_water = open(water_list_file,'w')
    # f_bridge = open(bridge_list_file,'w')

    # for img in lists_1[0]:
    #     f_background.write(img+'/n');

    # for img in lists_1[1]:
    #     f_ground.write(img+'/n');
    # for img in lists_1[2]:
    #     f_tree.write(img+'/n');
    # for img in lists_1[3]:
    #     f_roof.write(img+'/n');
    # for img in lists_1[4]:
    #     f_water.write(img+'/n');
    # for img in lists_1[5]:
    #     f_bridge.write(img+'/n');

# ###########################
#     ground_list_file=os.path.join(out_folder,'ground_list_no.txt')
#     tree_list_file=os.path.join(out_folder,'tree_list_no.txt')
#     roof_list_file=os.path.join(out_folder,'roof_list_no.txt')
#     water_list_file=os.path.join(out_folder,'water_list_no.txt')
#     bridge_list_file=os.path.join(out_folder,'bridge_list_no.txt')

#     f_ground = open(ground_list_file,'w')
#     f_tree = open(tree_list_file,'w')
#     f_roof = open(roof_list_file,'w')
#     f_water = open(water_list_file,'w')
#     f_bridge = open(bridge_list_file,'w')

#     for img in lists_2[0]:
#         f_ground.write(img+'/n');
#     for img in lists_2[1]:
#         f_tree.write(img+'/n');
#     for img in lists_2[2]:
#         f_roof.write(img+'/n');
#     for img in lists_2[3]:
#         f_water.write(img+'/n');
#     for img in lists_2[4]:
#         f_bridge.write(img+'/n');
#         ##do data balance. 
#     f_ground.close()


def PreAug(test_image): 
    img_aug=[] 
    if len(test_image.shape)>4:
        for i in range(len(test_image)):
            img_data=test_image[i]
            for a_id in range (4):
                img_a=np.rot90(img_data,4-a_id)
                img_aug.append(img_a)
    else:
        for a_id in range (4):
            img_a=np.rot90(test_image,4-a_id)
            img_aug.append(img_a)
    return img_aug

def PreAugBack(test_image): 
    img_back=[]
    if len(test_image.shape)>4:
        for id in range(len(test_image)):
            img=test_image[id]
            i=id%4
            if i==0:
                img_back.append(img)
            else:
                for mmm in range (i):
                    img=np.squeeze(img)
                    img=np.rot90(img)
                    img = np.expand_dims(img, axis=0)
                img_back.append(img)
    else:
        img_back.append(test_image)
    return img_back


if __name__ == '__main__':
    
    # img_folder='G:/DataSet/NUS_Facade/new_data_r'
    # label_folder='G:/DataSet/NUS_Facade/new_data_label'
    # out_folder='G:/DataSet/NUS_Facade/normalized_patchs'G:/DataSet/GRSS2019Contest/Track1-RGBGRSS2019Contest/Train-Track1-Truth/Track1-Truth
    # img_folder='G:/DataSet/GRSS2019Contest/Track1-RGB'
    # label_folder='G:/DataSet/GRSS2019Contest/Train-Track1-Truth/Track1-Truth'
    # out_folder='G:/DataSet/GRSS2019Contest/Track1-RGB_pathces_512/'
    # path_size=(512,512)
    # overlap_ratio=0
    # dataAugument(img_folder,label_folder,out_folder,path_size,overlap_ratio)
    #######
    img_folder=r'G:\DataSet\BuildingFacade\CMP\CMP_facade_DB_base\base'
    label_folder=r'G:\DataSet\BuildingFacade\CMP\CMP_facade_DB_base\labels'
    out_folder='G:/DataSet/CMP_Facade/train_data/512_patchs'
    path_size=(512,512)
    overlap_ratio=0.5
    DataSampleAnalysis(img_folder,label_folder,out_folder,path_size,overlap_ratio)
    ###################
    # label_folder='C:/TrainData/Track1/Train/Track1-Truth/'
    # AnalyzCategoryHeight(label_folder)
    #############
    # label_folder='C:/TrainData/Track1/Train/Track1-Truth/'
    # msi_folder='C:/TrainData/Track1/Train/orthos/'
    # GenerateMSIdata(msi_folder,label_folder,out_folder,path_size,overlap_ratio)
    ###################
    # img_folder='C:/TrainData/Track1/train/imgs'
    # pre_folder='G:/programs/dfc2019-master/track1/data/validate/track1-unet_rgb_h-evenloss-retrain-20190322e06-forval'
    
    # truth_folder='C:/TrainData/Track1/train/Track1-Truth'
    # label_folder='C:/TrainData/Track1/train/Track1-Truth'
    # class_test_folder='C:/TrainData/Track1/train/Track1-Truth'
    # #HeightScale(img_folder,label_folder,truth_folder,pre_folder)


    # from track1_metrics import compute_metrics_height
    # compute_metrics_height(pre_folder,class_test_folder,truth_folder)
    #HardSample('C:/TrainData/Track1/train/patch_512/img_patch','C:/TrainData/Track1/train/patch_512/label_patch', 'C:/TrainData/Track1/train/resized_512-/class_patch')