import json

import glob
import numpy as np

import os
import cv2
import params
import tifffile



TRAIN_task='CLS'
EXTRA_format='h'
EXTRA_data=False

def load_img(imgPath):
    """
    Load image
    :param imgPath: path of the image to load
    :return: numpy array of the image
    """
    if imgPath.endswith('.tif'):
        img = tifffile.imread(imgPath)
    else:
        raise ValueError('Install pillow and uncomment line in load_img')
#        img = np.array(Image.open(imgPath))
    return img
def AnalyzSampleCategory(label_folder):
    lists_1=[]
    lists_2=[]
    count=0
    for filename in os.listdir(label_folder):    ##JAX_004_007_CLS  JAX_004_007_RGB
        file_apx=filename[-3:]
        if file_apx=='jpg' or  file_apx=='png' or  file_apx=='tif':
            label=cv2.imread(os.path.join(label_folder,filename),0)
           # label=convertLas2Train(label, params.LABEL_MAPPING_LAS2TRAIN)
            img_size=label.shape[0]*label.shape[1]
            count=count+1
            mask=(label==1)
            y_new = label[mask]
            aa=y_new.size/img_size
            if aa>0.05:
                lists_1.append(str(count))
            else:
                lists_2.append(str(count))

    tree_list_file=os.path.join(label_folder,'small_tree_list_file.txt')
    roof_list_file=os.path.join(label_folder,'small_tree_list_file_no.txt')

    f_tree = open(tree_list_file,'w')
    f_roof = open(roof_list_file,'w')


    for img in lists_1:
        f_tree.write(img+'\n');
    f_tree.close()
    for img in lists_2:
        f_roof.write(img+'\n');


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
if __name__ == '__main__':
   
    label_folder='C:/TrainData/Track1/train/patch_512/label_patch_tree'
    AnalyzSampleCategory(label_folder)


