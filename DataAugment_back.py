import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
import params
from osgeo import gdal, gdalconst
import cv2
import os
import random
from keras.utils.np_utils import to_categorical

ia.seed(1)
CONTRAST_RANGE=(0.8, 2.0)
PERSPECTIVE_RANGE=(0.05, 0.09)


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

def normalize_image_to_path(img_path,label_path,patch_size,overlap_ratio,work_region=[],resize2size=[],convertLab=True,pad_edge=1,normalize_dsm=1, image_order=1):
    """
    this is used to crop image into small pathes. 
    img_path: the path of the image file.
    label_path: the path of the label file. The label can be dsm.
    patch_size: the size of the small pathes.
    overlap_ratio: the overlap of each patch.
    work_region: the region that only crop from. The default is the whole image.
    resize2size: resize the image first. The default is no resize.
    convertLab: if to convert the label to taining (0-4) or submitting(2-17).
    pad_edge: the way of crop images at the boundary, it doesn't work here. 
    normalize_dsm: if to subtract the miminal value for the dsm data.
    image_order: the channl first(1) or channle last(2).(gdal is channel first)
    """
    if 1:#img_path[:-3]=='tif'):
        gd_reader = GRID()
        proj,geotrans,img = gd_reader.read_img(img_path) 
        proj,geotrans,label = gd_reader.read_img(label_path) 
    else:
        img=img_path
        label=label_path
    if len(work_region)>0:
        img = img[:,work_region[0]:work_region[1],work_region[2]:work_region[3]]
        if (label.shape[0]>work_region[1]-work_region[0] or label.shape[1]>work_region[3]-work_region[2]):
            label=label[:,work_region[0]:work_region[1],work_region[2]:work_region[3]]

    if len(resize2size)>0:
        img = img.transpose(1,2,0)
        img=cv2.resize(img,resize2size)
        img=img.transpose(2,0,1)
        label=cv2.resize(label,resize2size,cv2.INTER_NEAREST)
    

    if convertLab:
        label=convertLas2Train(label, params.LABEL_MAPPING_LAS2TRAIN)
    elif normalize_dsm:
        nan_data=np.isnan(label)
        label[nan_data] = 99999
        min_t=label.min()
        label=label-min_t
        label[nan_data]=0
    if image_order==0:
        rows=img.shape[0]
        cols=img.shape[1]
    elif image_order==1:
        rows=img.shape[1]
        cols=img.shape[2]

    imgs=[]
    labels=[]

    if img.shape[1]<=patch_size[0] and img.shape[2]<=patch_size[1]:
        imgs.append(img)
        labels.append(label)
    else:
#        patch_ranges=calculate_cut_range(img.shape[1:3], patch_size=[patch_size[0],patch_size[1]],overlap=overlap_ratio)


        patch_height = patch_size[0]
        patch_width = patch_size[1]
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
                label_patch=label[int(y_s):int(y_e),int(x_s):int(x_e)]
                imgs.append(img_patch);
                labels.append(label_patch)
    return imgs,labels,[rows,cols]
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
            line = line.strip('\n')           
            img_files.append(line)
####find labels
    file=os.path.join(datafolder,'label_list.txt')
    fp = open(file)
    lines = fp.readlines()
    fp.close()
    for line in lines:
            line = line.strip('\n')           
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
            f_img.write('img_patch/'+img_write_path+'\n');
            f_label.write('label_patch/'+label_write_path+'\n');

    f_img.close()
    f_label.close()

def dataAugument(img_folder, label_folder, out_folder,path_size,overlap_ratio):
#    crop_normalized_path(img_folder,label_folder,out_folder,path_size,overlap_ratio,)
    imageAugment(out_folder)

def crop_normalized_patch_track1(img_folder, label_folder,out_folder,path_size,overlap_ratio):
    """
    this is used to crop input images into differnt patchese.
    img_folder: the folder contains the image need to be cropped
    label_folder: the folder contains the labels. The label can also be dsm and data with out format.
    out_folder: the out folder that contains the cropped data.
    patch_size: the patch size that need to be cropped.
    overlap: the overlap between each patch. range(0,1).    
    """
    if os.path.exists(out_folder)==0:
        os.makedirs(out_folder)
    sub_img_folder=os.path.join(out_folder,'img_patch')
    if os.path.exists(sub_img_folder)==0:
        os.makedirs(sub_img_folder)
    sub_label_folder=os.path.join(out_folder,'dsm_patch_p')#
    if os.path.exists(sub_label_folder)==0:
        os.makedirs(sub_label_folder)
    img_list_file=os.path.join(out_folder,'img_list.txt')
    label_list_file=os.path.join(out_folder,'dsm_p_list.txt')#
    f_img = open(img_list_file,'w')
    f_label = open(label_list_file,'w')
    gd_reader = GRID()
    for filename in os.listdir(img_folder):    ##JAX_004_007_CLS  JAX_004_007_RGB
        file_apx=filename[-3:]
        if file_apx=='jpg' or  file_apx=='png' or  file_apx=='tif':
            label_path=os.path.join(label_folder , filename[:-7]+'AGL.tif')
            img_path=os.path.join(img_folder,filename)
            if os.path.exists(label_path):
                [imgs,labels,size_o]=normalize_image_to_path(img_path,label_path,path_size,overlap_ratio,convertLab=False,pad_edge=1,normalize_dsm=0,image_order=1)
                for i in range(len(imgs)):
                    img_write_path=sub_img_folder+'/'+filename[:-4]+'_'+str(i)+'.tif'
                    label_write_path=sub_label_folder+'/'+filename[:-8]+'_AGL_'+str(i)+'.tif'#
               #     gd_reader.write_img(img_write_path,[],[],imgs[i]) #写数据
                    gd_reader.write_img(label_write_path,[],[],labels[i]) #写数据
                    f_img.write(filename[:-4]+'_'+str(i)+'.tif'+'\n');
                    f_label.write(filename[:-8]+'_AGL_'+str(i)+'.tif'+'\n');#
    f_img.close()
    f_label.close()

def AnalyzSampleCategory(label_folder):
    lists_1=[[],[],[],[],[]]
    lists_2=[[],[],[],[],[]]
    ratios=[0.2,0.1,0.1,0.1,0.1]
    
    for filename in os.listdir(label_folder):    ##JAX_004_007_CLS  JAX_004_007_RGB
        file_apx=filename[-3:]
        if file_apx=='jpg' or  file_apx=='png' or  file_apx=='tif':
            label=cv2.imread(os.path.join(label_folder,filename),0)
            img_size=label.shape[0]*label.shape[1]
            for i in range(5):
                mask=(label==i)
                y_new = label[mask]
                aa=y_new.size/img_size
                if aa>ratios[i]:
                    lists_1[i].append(filename)
                else:
                    lists_2[i].append(filename)
    
    return lists_1,lists_2





    
def DataSampleAnalysis(img_folder,label_folder,out_folder,patch_size,overlap_ratio):
    """
    this code is used to crop the large training data into smaller patches and analyse their numbers of samples in each class.
    img_folder: the folder contains the large images.
    label_folder: the folder contains the labels. The label can also be dsm and data with out format.
    out_folder: the out folder that contains the cropped data.
    patch_size: the patch size that need to be cropped.
    overlap: the overlap between each patch. range(0,1).
    """
    driver = gdal.GetDriverByName('HFA')
    driver.Register()
    number_classes=5
    crop_normalized_patch_track1(img_folder,label_folder,out_folder,patch_size,overlap_ratio)
    patch_lable=os.path.join(out_folder,'label_patch/')
    lists_1,lists_2=AnalyzSampleCategory(patch_lable)
    num_samples=len(lists_1[0])+len(lists_2[0])
    ratios=[]
    for i in range(number_classes):
        ratios.append(len(lists_1[i])/num_samples)
    ground_list_file=os.path.join(out_folder,'ground_list.txt')
    tree_list_file=os.path.join(out_folder,'tree_list.txt')
    roof_list_file=os.path.join(out_folder,'roof_list.txt')
    water_list_file=os.path.join(out_folder,'water_list.txt')
    bridge_list_file=os.path.join(out_folder,'bridge_list.txt')

    f_ground = open(ground_list_file,'w')
    f_tree = open(tree_list_file,'w')
    f_roof = open(roof_list_file,'w')
    f_water = open(water_list_file,'w')
    f_bridge = open(bridge_list_file,'w')

    for img in lists_1[0]:
        f_ground.write(img+'\n');
    for img in lists_1[1]:
        f_tree.write(img+'\n');
    for img in lists_1[2]:
        f_roof.write(img+'\n');
    for img in lists_1[3]:
        f_water.write(img+'\n');
    for img in lists_1[4]:
        f_bridge.write(img+'\n');
###########################
    ground_list_file=os.path.join(out_folder,'ground_list_no.txt')
    tree_list_file=os.path.join(out_folder,'tree_list_no.txt')
    roof_list_file=os.path.join(out_folder,'roof_list_no.txt')
    water_list_file=os.path.join(out_folder,'water_list_no.txt')
    bridge_list_file=os.path.join(out_folder,'bridge_list_no.txt')

    f_ground = open(ground_list_file,'w')
    f_tree = open(tree_list_file,'w')
    f_roof = open(roof_list_file,'w')
    f_water = open(water_list_file,'w')
    f_bridge = open(bridge_list_file,'w')

    for img in lists_2[0]:
        f_ground.write(img+'\n');
    for img in lists_2[1]:
        f_tree.write(img+'\n');
    for img in lists_2[2]:
        f_roof.write(img+'\n');
    for img in lists_2[3]:
        f_water.write(img+'\n');
    for img in lists_2[4]:
        f_bridge.write(img+'\n');
        ##do data balance. 

if __name__ == '__main__':
    
    # img_folder='G:/DataSet/NUS_Facade/new_data_r'
    # label_folder='G:/DataSet/NUS_Facade/new_data_label'
    # out_folder='G:/DataSet/NUS_Facade/normalized_patchs'G:\DataSet\GRSS2019Contest\Track1-RGBGRSS2019Contest\Train-Track1-Truth\Track1-Truth
    # img_folder='G:/DataSet/GRSS2019Contest/Track1-RGB'
    # label_folder='G:/DataSet/GRSS2019Contest/Train-Track1-Truth/Track1-Truth'
    # out_folder='G:/DataSet/GRSS2019Contest/Track1-RGB_pathces_512/'
    # path_size=(512,512)
    # overlap_ratio=0
    # dataAugument(img_folder,label_folder,out_folder,path_size,overlap_ratio)

    img_folder='C:/TrainData/Track1/Train/Track1-RGB/'
    label_folder='C:/TrainData/Track1/Train/Track1-height-p/'
    out_folder='C:/TrainData/Track1/Train/patch_512/'
    patch_size=(512,512)
    overlap_ratio=0.5
    DataSampleAnalysis(img_folder,label_folder,out_folder,patch_size,overlap_ratio)
