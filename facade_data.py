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
import glob

from DataAugment import calculate_cut_range
from track3_data import load_all_data_files,input_generator_RGBH, weighted_categorical_crossentropy
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

def input_generator(img_files, label_files, batch_size, num_category=params.NUM_CATEGORIES):

    N = len(img_files) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

    #executor = ProcessPoolExecutor(max_workers=3)

    while True:
        for inds in batchInds:
            img_batch = [img_files[ind] for ind in inds]
            label_batch = [label_files[ind] for ind in inds]

            imgdata, gts= load_cnn_batch(img_batch, label_batch)#, executor)      
            
            if 0:
                import matplotlib.pyplot as plt 
                plt.subplot(221) #用于显示多个子图121代表行、列、位置
                plt.imshow(imgdata[0,:,:,0:3])
                plt.title('org')
                plt.subplot(222)
                plt.imshow(gts[0,:,:,0])
                plt.title('background') #添加标题
                plt.subplot(223)
                plt.imshow(imgdata[1,:,:])
                plt.title('dsm') #添加标题
                plt.subplot(224)
                plt.imshow(gts[1,:,:,0])
                plt.title('roof') #添加标题
                plt.show()
            yield (imgdata, gts)
           # return (imgdata, gts)

def load_cnn_batch(img_batch, label_batch):
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

    inputs=[]
    labels=[]
    img_data=load_img(rgb_file)
    label_data=load_img(gts_file)
    if label_data.shape[2]>1:
        label_data=label_data[:,:,0]
    img_data=img_data.astype(np.float)
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
                plt.imshow(img_data/128)
                plt.title('org')
                plt.subplot(122)
                plt.imshow(label_data)
                plt.title('label_data') #添加标题
                plt.show()
            
    currLabel=np.array(label_data,np.float)
    currLabel = to_categorical(currLabel, num_classes=int(params.NUM_CATEGORIES)+1)
    label_data =currLabel[:,:,0:-1]
    from dataFunctions import image_augmentation,image_augmentation_test
    imageMedium,labelMedium = image_augmentation_test(img_data, label_data)
    
    inputs.append(img_data)
    labels.append(label_data)    
    inputs.append(imageMedium)
    labels.append(labelMedium)
    return inputs, labels
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



def GetAllPossibelColours(data_folder_id):
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

def GenerateNumbreLabels(data_folder_id,out_folder):
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
    for i in range(len(img_lines)):
        label_file=label_lines[i].strip('\n')
        label_name=os.path.split(label_file)[-1]
        new_lalel_path=os.path.join(out_folder,label_name)
        label_color=cv2.imread(os.path.join(img_folder,label_file))
        label_color=cv2.cvtColor(label_color, cv2.COLOR_BGR2RGB)
        new_label=np.zeros((label_color.shape[0], label_color.shape[1]),dtype=int)
        for i in range(len(params.TRAIN2SHOW)):
            color=np.array(params.TRAIN2SHOW[i])
            mask=(label_color[:,:,0]==color[0]) & (label_color[:,:,1]==color[1]) & (label_color[:,:,2]==color[2])
           # mask=(label_color[:,:,0]==color[0] and label_color[:,:,1]==color[1] and label_color[:,:,2]==color[2])
            new_label[mask]=i

        cv2.imwrite(new_lalel_path, new_label)
        
        # for y in range(label_color.shape[0]):
        #     for x in range(label_color.shape[1]):
        #         current_color=label_color[y,x,:]
        #         for i in range(len(params.TRAIN2SHOW)):
        #             color=np.array(params.TRAIN2SHOW[i])
        #             if color.all()==current_color.all():
        #                 new_label[y,x]=i
        # cv2.imwrite(new_lalel_path, new_label)

def NormalizeImage(img_folder):
    '''
    normalize/cut the image into 512*512
    '''
    
def load_all_data_files_balanced_patches(data_folder,label_folder='',text_files=[],vali_ratio=0.1,max_samples=-1,net_name='unet_rgb_c'):
    
    if len(label_folder)<3:
        class_folder='label_patch'
    else:
        class_folder=label_folder
    
    img_folder='img_patch'
    class_file_foler='class_file_record'

    #balanced_sample_number=1600
    imgs = []
    gts=[]
    extras=[]

    imgs_v=[]
    gts_v=[]
    extras_v=[]
    label_list_path=os.path.join(data_folder,'label_list.txt')
    img_list_path=os.path.join(data_folder,'img_list.txt')
    img_files=[]
    fp = open(img_list_path)
    lines = fp.readlines()
    fp.close()
    for line in lines:
        line = line.strip('\n')
        img_files.append(line)
    if len(text_files)<1: ##load all txt recording all the images.
        glob_path=os.path.join(data_folder,class_file_foler,'*.txt')
        files=glob.glob(glob_path)
        for txt in files:
            text_files.append(txt)
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
    
    val_ids=[]
    for i in range(len(text_files)):
        class_ids=all_ids[i]
        val_num=int(len(class_ids)*vali_ratio)
        idx = np.random.permutation(len(class_ids))
        ids=idx[0:val_num]
        val_ids.extend(class_ids[ind] for ind in ids)
    train_ids=[]
    batch_star=[0]*len(clasee_samples)
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
        batch_ids=set(batch_ids)-set(val_ids)
        train_ids.extend(batch_ids)
        record_sampels=record_sampels+min_samples

    for id in train_ids:
        line=img_files[id-1]
        gts.append(os.path.join(data_folder,class_folder,line.replace('jpg','png')))
        imgs.append(os.path.join(data_folder,img_folder,line))

    for id in val_ids:
        line=img_files[id-1]
        gts_v.append(os.path.join(data_folder,class_folder,line.replace('jpg','png')))
        imgs_v.append(os.path.join(data_folder,img_folder,line))

    return imgs, gts, imgs_v,  gts_v
def load_all_data_test(data_folder):
    imgs=[]
    pathes=[]
    
    #img_folder=os.path.join(data_folder,test_img_folder)
    img_files = glob.glob(os.path.join(data_folder, '*.jpg'))

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

def GetPredictData(ortho_path, channels,extra_path='',path_size=(256,256),overlap_ratio=0.5,convertLab=False,normalize_dsm=0,img_ioa=[],resize2size=[]):
    from scipy import ndimage
    import matplotlib.pyplot as plt
    img_aug=[]
    from DataAugment import normalize_image_to_path,getAuger_p_p
    #label_input=False
    [imgs,dsms,image_size]=normalize_image_to_path(ortho_path,'',path_size,overlap_ratio)

    for i in range(len(imgs)):
        img_data=imgs[i]
        #img_data = img_data.transpose(1,2,0)
        img_data=img_data.astype(np.float32)
        img_data=img_data/125.0-1
        img_aug.append(img_data)
        img_a=cv2.flip(img_data,1)
        img_aug.append(img_a)
        if 0:
            plt.subplot(121) #用于显示多个子图121代表行、列、位置
            plt.imshow(img_data)
            plt.title('org')
            plt.subplot(122)
            plt.imshow(img_a)
            plt.title('flipped') #添加标题
            plt.show()
    return img_aug,image_size
def goback_auger_p(imgs):
    img_back=[]
    #ia.seed(1)
    contrast_range=(0.0,0.0)
    for id in range(len(imgs)):
        img=imgs[id]
        i=id%2
        if i==0:
            img_back.append(img)
        else:
            for mmm in range (i):
                img=np.squeeze(img)
                img2=cv2.flip(img,1)
               # img2 = np.expand_dims(img2, axis=0)
#            auger=getAuger_p_p(4-i)
#            img_a=auger.augment_image(img)
                if 0:
                    plt.subplot(121) #用于显示多个子图121代表行、列、位置
                    plt.imshow(img)
                    plt.title('org')
                    plt.subplot(122)
                    plt.imshow(img2)
                    plt.title('flipped') #添加标题
                    plt.show()
                img_back.append(img2)
    return img_back
def convert2colors(pred):
    new_label=np.zeros((pred.shape[0], pred.shape[1],3))
    for i in range(len(params.TRAIN2SHOW)):
        mask=pred==i
        color=np.array(params.TRAIN2SHOW[i])
        r=new_label[:,:,0]
        g=new_label[:,:,1]
        b=new_label[:,:,2]
        r[mask]=color[0]
        g[mask]=color[1]
        b[mask]=color[2]
        new_label[:,:,0]=r
        new_label[:,:,1]=g
        new_label[:,:,2]=b
        new_label=new_label.astype('uint8')
    return new_label

def patch2img(patches,img_size,patch_weights,num_class=5, overlap=0.5):
    patches=np.squeeze(patches)
    patch_wid=patches[0].shape[1]
    patch_hei=patches[0].shape[0]
    vote=np.zeros((img_size[0],img_size[1],num_class))
    
    if patch_weights.shape[0]!=patch_hei or patch_weights.shape[1]!=patch_wid:
        new_patches=[]
        for i in range(len(patches)):
            new_patches.append(cv2.resize(patches[i],(patch_weights.shape[1],patch_weights.shape[0]),cv2.INTER_NEAREST))
        patches=new_patches

    patch_ranges=calculate_cut_range(img_size, patch_size=[patch_hei,patch_wid],overlap=overlap)

    N=len(patches)
    idx = range(N)
    patch_weights = patch_weights[:,:,np.newaxis]
    batchInds = get_batch_inds(2, idx, N) 
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

    pred = np.argmax(vote, axis=-1).astype('uint8')
    return pred

if __name__ == '__main__':
    outfolder=r'G:\DataSet\BuildingFacade\CMP\CMP_facade_DB_base\labels'
    GenerateNumbreLabels(0,outfolder)
    #DATA_NAMES[0]='CMP'
    #DATA_NAMES[1]='ParisFacades'
    data_id=0
 #   get_all_train_files(data_id)    
    GetAllPossibelColours(data_id)