#this is used to deal with facade data
import numpy as np
import glob
from segmentation_models import UnetRegressor,Unet,pspnet#PSPNet
#from segmentation_models import psp_50
from dataFunctions import *

from keras.callbacks import ModelCheckpoint
from keras.callbacks import *
from keras.layers import Input
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam, SGD
from tqdm import tqdm
import json
import tensorflow as tf
import facade_params as params
import cv2
from track3_data import load_all_data_files,input_generator_RGBH, weighted_categorical_crossentropy
def Merge_track3_results(self, result_folder,out_folder,offset_folder):
    site_images=[]
    site_names=[]
    if os.path.exists(out_folder)==0:
        os.makedirs(out_folder)
    else:
        glob_path=os.path.join(out_folder,'*.tif')
        files=glob(glob_path) 
        for file in files:
            os.remove(file)
    glob_path=os.path.join(result_folder,'*.tif')
    files=glob(glob_path)
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
    for m in range(len(site_names)):
        imgs=site_images[m]
        im=cv2.imread(imgs[0],0)
        vote_map=np.zeros((im.shape[0],im.shape[1],params.NUM_CATEGORIES))
        for img_p in imgs:
            im=cv2.imread(img_p,0)
            one_hot=to_categorical(im,params.NUM_CATEGORIES)
            for i in range(vote_map.shape[-1]):
                vote_map[:,:,i]=vote_map[:,:,i]+one_hot[:,:,i]
        pred=np.argmax(vote_map,axis=-1).astype('uint8')
        if pred.shape[0]>512 or pred.shape[1]>512:
            offset_file=os.path.join(offset_folder,site_names[m]+'_DSM.txt')
            offset = np.loadtxt(offset_file)
            offset=offset.astype('int')
            pred=pred[offset[1]:offset[1]+512,offset[0]:offset[0]+512]
        pred=convert_labels(pred,params,toLasStandard=True)
        out_path=os.path.join(out_folder,site_names[m]+'_CLS.tif')
        tifffile.imsave(out_path,pred,compress=6)

def Merge_results_with_baseline(self,result_folder,baseline_folder):
    from dataFunctions import convert_labels
    glob_path=os.path.join(result_folder,'*CLS.tif')
    files=glob(glob_path)
    
    for img in files:
        image_name=os.path.split(img)[-1]
        site_name=image_name[0:7]
        result_my=cv2.imread(img,0)
        result_my=convert_labels(result_my,params,toLasStandard=False)
        result_baseline=cv2.imread(os.path.join(baseline_folder,image_name),0)
        result_baseline=convert_labels(result_baseline,params,toLasStandard=False)
        result_merge=np.zeros([result_baseline.shape[0],result_baseline.shape[1]], dtype = 'uint8')
#           result_merge[(result_baseline==0)]=result_baseline[(result_baseline==0)]
#            result_merge[(result_baseline==2)]=result_baseline[(result_baseline==2)]
        result_merge[(result_my==2)]=result_my[(result_my==2)]
        result_merge[(result_baseline==1)]=result_baseline[(result_baseline==1)]
        result_merge[(result_my==1)]=result_my[(result_my==1)]
        result_merge[(result_baseline==3)]=result_baseline[(result_baseline==3)]
        #result_merge[(result_my==3)]=result_my[(result_my==3)]
        result_merge[(result_baseline==4)]=result_baseline[(result_baseline==4)]
        result_merge[(result_my==4)]=result_my[(result_my==4)]
        result_merge=convert_labels(result_merge,params,toLasStandard=True)
        out_path=os.path.join(result_folder,site_name+'_CLS_Merge.tif')
        tifffile.imsave(out_path,result_merge,compress=6)
def ConvertLabel(input_label,direction='show2train'):
    
    if direction=='show2train':
        if len(input_label.shape())<3:
            print('the input shape {shape} is not work with the direction {direc}:'
            .format(shape=input_label.shape[-1],direc=direction ))
            return
        output=np.empty((input_label.shape[0:2]))

        from facade_params import TRAIN2SHOW as tabel 


    elif direction=='train2show':
        if len(input_label.shape())>1:
            print('the input shape {shape} is not work with the direction {direc}:'
            .format(shape=input_label.shape[-1],direc=direction ))
            return        
        output=np.empty((input_label[0,2],3))
        from facade_params import SHOW2TRAIN as tabel

    
    return output

def load_all_train_data(data_folder_id):
    data_name=params.DATA_NAMES[data_folder_id]
    data_folder=params.DATA_FOLDERS[data_folder_id]
    if data_name=='CMP':
        img_folder=os.path.join(data_folder,'base')

    ortho_list_file=os.path.join(data_folder,'img_list.txt')
    dsm_list_file=os.path.join(data_folder,'label_list.txt')

    f_ortho = open(ortho_list_file,'r')
    f_dsm = open(dsm_list_file,'r')
    img_lines = f_ortho.readlines()
    label_lines=f_dsm.readlines()
    f_ortho.close()
    f_dsm.close()

    img_train=[]
    label_train=[]
    for i in range(len(img_lines)):
        img_file=img_lines[i].strip('\n')
        label_file=label_lines[i].strip('\n')
        img=cv2.imread(os.path.join(img_folder,img_file))
        label_color=cv2.imread(os.path.join(img_folder,label_file))
        label=ConvertLabel(label_color,direction='show2train')
        img_train.append(img)
        label_train.append(label)
   
    img_train = np.array(img_train, np.float32)
    img_train=img_train/125.0-1
    return img_train, label_train





def get_all_train_files(data_folder_id):
    imgs=[]
    dsms=[]
    data_name=params.DATA_NAMES[data_folder_id]
    data_folder=params.DATA_FOLDERS[data_folder_id]
    if data_name=='CMP':
        img_folder=os.path.join(data_folder,'base')
    img_files = glob(os.path.join(img_folder, '*.jpg'))
    for imgPath in img_files:
        imageName = os.path.split(imgPath)[-1]
        labelName=imageName[0:-3]+'png'
        imgs.append(imageName)
        dsms.append(labelName)



    ###########################
    ortho_list_file=os.path.join(data_folder,'img_list.txt')
    dsm_list_file=os.path.join(data_folder,'label_list.txt')

    f_ortho = open(ortho_list_file,'w')
    f_dsm = open(dsm_list_file,'w')

    for i in range(len(imgs)):
        f_ortho.write(imgs[i]+'\n');
        f_dsm.write(dsms[i]+'\n');
    f_ortho.close()
    f_dsm.close()



def load_all_train_data_test(data_folder_id):
    data_name=params.DATA_NAMES[data_folder_id]
    data_folder=params.DATA_FOLDERS[data_folder_id]
    if data_name=='CMP':
        img_folder=os.path.join(data_folder,'base')

    ortho_list_file=os.path.join(data_folder,'img_list.txt')
    dsm_list_file=os.path.join(data_folder,'label_list.txt')

    f_ortho = open(ortho_list_file,'r')
    f_dsm = open(dsm_list_file,'r')
    img_lines = f_ortho.readlines()
    label_lines=f_dsm.readlines()
    f_ortho.close()
    f_dsm.close()

    img_train=[]
    label_train=[]
    colors=[]
    num_colors=[]
    for i in range(len(img_lines)):
        img_file=img_lines[i].strip('\n')
        label_file=label_lines[i].strip('\n')
        img=cv2.imread(os.path.join(img_folder,img_file))
        label_color=cv2.imread(os.path.join(img_folder,label_file))
        for y in range(label_color.shape[0]):
            for x in range(label_color.shape[1]):
                if len(colors)==0:
                    colors.append(label_color[y,x,:])
                    num_colors.append(1)
                else:
                    new_color=True
                    for m in range (len(colors)):
                        aa=colors[m]
                        current_color=label_color[y,x,:]
                        if (aa==current_color).all():
                            new_color=False
                            num_colors[m]=num_colors[m]+1
                    if new_color:
                        colors.append(label_color[y,x,:])
                        num_colors.append(1)
    return colors,num_colors




if __name__ == '__main__':
    #DATA_NAMES[0]='CMP'
    #DATA_NAMES[1]='ParisFacades'
    data_id=0
 #   get_all_train_files(data_id)    
    load_all_train_data_test(data_id)