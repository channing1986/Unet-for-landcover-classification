__author__ = 'Changlin'
__version__ = 0.1

import numpy as np
import math
from dataFunctions import *
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.callbacks import *
from keras.layers import Input
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam, SGD
from tqdm import tqdm
import json
import tensorflow as tf
import params
#from scipy.special import softmax
from grss_data import input_generator,get_batch_inds
from grss_models import get_model
#from track1_data import load_all_data_files,input_load_train_data,input_generator_online_process,input_generator_RGB
from grss_data import load_all_data_files_balanced_patches,load_all_data_test,load_all_data_files

class Class_net:
    def __init__(self, config=None, mode=None):
        """
        Initializes baseline by setting config, mode, and loading mean values to be used in the case non-RGB imagery is being used
        :param config: input parameters from config.py
        :param mode: either SEMANTIC_MODE or SINGLEVIEW_MODE from config (0 or 1)
        :return: None
        """
        self.config=config
        self.mode = mode
        self.meanVals = None
        if config.NUM_CHANNELS != 3:
            if config.MEAN_VALS_FILE is None:
                self.meanVals = np.zeros(config.NUM_CHANNELS).tolist()
            else:
                self.meanVals = json.load(open(config.MEAN_VALS_FILE))

        # TensorFlow allocates all GPU memory up front by default, so turn that off

        os.environ["CUDA_VISIBLE_DEVICES"]=config.GPUS
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=tf_config))
   


    def train(self,data_folder,weighted_loss=False,patch_balance=False,pre_trained_weight_path='',net_name='unet_rgb_c'):
        num_class=self.config.NUM_CATEGORIES
        check_folder=os.path.join(params.CHECKPOINT_DIR,net_name)
        if os.path.exists(check_folder)==0:
            os.mkdir(check_folder)
        CHECKPOINT_DIR=check_folder
        CHECKPOINT_PATH = os.path.join(check_folder, 'weights.{epoch:02d}.hdf5')

        ## load data
        input_shape=[512,512,3]
        val_ratio=0.1
        label_folder='label_patch'
        if patch_balance:
            img_train, lable_train, img_val, label_val=load_all_data_files_balanced_patches(data_folder,
                                                        vali_ratio=val_ratio)
        else:
            img_train, lable_train, img_val, label_val=load_all_data_files(data_folder,vali_ratio=val_ratio)
        
        batch_size=self.config.BATCH_SZ

        train_generator=input_generator(img_train, lable_train, batch_size,net_name=net_name,num_category=num_class)
        val_generator = input_generator(img_val,label_val,batch_size,net_name=net_name,num_category=num_class)

        ## prepare training parameters
        nb_epoch=200
        num_training_sample=len(img_train)
        n_batch_per_epoch=num_training_sample//batch_size
        num_val_sample=len(img_val)
        n_batch_per_epoch_val=num_val_sample//batch_size
        csv_logger = CSVLogger(os.path.join(CHECKPOINT_DIR,'train.csv'))
        checkpoint = ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor='loss', verbose=1, save_best_only=False,
                                        save_weights_only=True, mode='auto', period=self.config.MODEL_SAVE_PERIOD)
        callbacks=[csv_logger,  checkpoint]

        ## get model and train
        model,input_shape=get_model(net_name,num_class,pre_trained_weight_path,input_shape=input_shape,weighted_loss=weighted_loss)
        model.fit_generator(train_generator, steps_per_epoch=n_batch_per_epoch,
                            validation_data=val_generator, validation_steps=n_batch_per_epoch_val,
                            epochs=nb_epoch, callbacks=callbacks)


    def test(self,data_folder,weight_path,pre_augment=False,pre_patch=False, results_folder='',net_name='',num_class=''):
        #from grss2019_data import load_all_data_test
        from DataAugment import Img2Patch,PreAugBack,PreAug,Patch2Img,GetPatchWeight
        if not num_class:
            num_class=self.config.NUM_CATEGORIES
        if not net_name:
            net_name='unet_rgb_c'
        if not results_folder:
            results_folder=os.path.join(params.OUTPUT_DIR,'_'+net_name+os.path.split(weight_path)[-1])        
        if os.path.exists(results_folder)==0:
            os.mkdir(results_folder)
        
        from dataFunctions import convert_labels
        
        #####
        extension='.tif'
        #img_test=load_all_data_test_val(data_folder,channels)
        img_test=load_all_data_test(data_folder,extension)

        if pre_patch:
            input_shape=self.config.PATCH_SZ
        else:
            input_shape=[None,None,3]

        model,input_shape=get_model(net_name,num_class,weight_path,input_shape=input_shape)
        extra_dsm=False
        extra_class=False
        convertLab=False

        if pre_patch:
            patch_weights=GetPatchWeight([input_shape[0],input_shape[1]],pad=64,last_value=0.05)
            
        batch_size=1
        overlap=0.00
        resize2size=()
        resized=False
        print('Number of files = ', len(img_test))
        path_size=([input_shape[0],input_shape[1]])

        for i in range(len(img_test)):
            print('Processing files:', i/len(img_test))
            ortho_path=img_test[i]
            predicted_batches=[]
            imageName = os.path.split(ortho_path)[-1]
            #outName = imageName.replace('RGB', "CLS")
            outName = imageName[:-4]+'_CLS'+imageName[-4:]
            test_image=load_img(ortho_path)
            if len(test_image.shape)<3:
                test_image=cv2.cvtColor(test_image, cv2.COLOR_BGRA2RGB)
            test_image=test_image/125.-1

            if pre_patch:
                test_image_patches=Img2Patch(test_image,patch_size=path_size,overlap_rati=overlap)
                N = len(test_image_patches)
            else:
                size_10=test_image.shape[0]
                size_20=test_image.shape[1]
                size_1=math.ceil(test_image.shape[0]/32)*32
                size_2=math.ceil(test_image.shape[1]/32)*32
                test_image_patches=test_image
                if size_1!=size_10 or size_2!=size_20:
                    test_image_patches=cv2.resize(test_image_patches, (size_1,size_2))
                test_image_patches=test_image_patches[np.newaxis,:,:,:]
                N=1
            if pre_augment:
                test_image_patches=PreAug(test_image_patches)
                N=N*4#total number of images
            idx = range(N)
            batchInds = get_batch_inds(batch_size, idx, N,predict=True) 
            for inds in range(len(batchInds)):
                img_batch = [test_image_patches[ind] for ind in batchInds[inds]]
                img_batch=np.array(img_batch,np.float32)
                pred = model.predict(img_batch)
                pred = np.argmax(pred, axis=-1).astype('uint8')
                predicted_batches.extend(pred)

            pred=predicted_batches
            if pre_augment:
                pred=PreAugBack(pred,num_class=num_class)
            if pre_patch:
                pred=Patch2Img(pred,test_image.shape,patch_weights,num_class=num_class, overlap=overlap)
                

            #pred=convert_labels(pred,self.config,toLasStandard=True)
            pred=np.squeeze(pred)
            if (not pre_patch) and (size_1!=size_10 or size_2!=size_20):
                pred=cv2.resize(pred, (size_10,size_20),interpolation=cv2.INTER_NEAREST)
            tifffile.imsave(os.path.join(results_folder, outName), pred, compress=6)

    

def train_net(data_folder):
    detector=Class_net(params)
    net_name='unet_rgb_c'
    pre_trained_weight_path=''
    weighted_loss=True
    patch_balance=False
    detector.train(data_folder=data_folder,weighted_loss=weighted_loss,patch_balance=patch_balance)

def test_net(data_folder):

    detector=Class_net(params)
    net_name='unet_rgb_c'
    weight_file='weights.66.hdf5'
    weight_path=os.path.join(params.CHECKPOINT_DIR,net_name,weight_file)
    results_folder=os.path.join(params.OUTPUT_DIR,net_name+weight_file)
    num_class=params.NUM_CATEGORIES
    ####data_folder,weight_path,pre_augment=False,pre_patch=False, results_folder='',net_name='',num_class=''
    pre_patch=True
    pre_augment=True
    detector.test(data_folder,weight_path,pre_augment=pre_augment,pre_patch=pre_patch)

if __name__ == '__main__':

    data_folder=r'C:\TianZhi2019\data'
    #train_net(data_folder)
    test_folder=r'G:\DataSet\TianZhi2019\src'
    test_net(test_folder)
    #test_net('C:/TrainData/Track1/train/')
    #test_net('C:/TrainData/Track1/Test-Track1',False)
 #   test_net('G:/shengxi',False)
    #rain_net('C:/TrainData/Track1/train/patch_512')
 #   test_single_classification()
 #   train_singel_class()
#     detector=Track3_net(params,0)
# #    check_folder= './checkpoint_track3_psp_1/'
# #    detector.train_track3('psp',check_folder)
#     #detector.train_track3_single()

#     ##############
#     results_folder='G:/programs/dfc2019-master/track1/data/validate/Track3-Submission-1_weighted_loss'
#     out_folder='G:/programs/dfc2019-master/track3/data/validate/Track3-Submission-1_weighted_loss'
#     baseline_folder='C:/TrainData/Track3/Test/track3_baseline_result'
#     offset_folder='C:/TrainData/Track3/Test_new/offset'
#     detector.Merge_track3_results(results_folder,out_folder,offset_folder)
#     detector.Merge_results_with_baseline(out_folder,baseline_folder)