__author__ = 'Changlin'
__version__ = 0.1

import numpy as np
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
            results_folder=os.path.join(params.OUTPUT_DIR,net_name+weight_path)        
        if os.path.exists(results_folder)==0:
            os.mkdir(results_folder)
        
        from dataFunctions import convert_labels
        
        #####
        channels=net_name.split('_')[1]
        #img_test=load_all_data_test_val(data_folder,channels)
        img_test=load_all_data_test(data_folder,channels)

        if pre_patch:
            input_shape=self.config.PATCH_SZ
        else:
            input_shape=[None,None,3]

        model,input_shape=get_model(net_name,num_class,weight_path,input_shape=input_shape)
        extra_dsm=False
        extra_class=False
        convertLab=False

        patch_weights=GetPatchWeight([input_shape[0],input_shape[1]],pad=64,last_value=0.05)
        batch_size=1
        overlap=0.00
        resize2size=()
            
        print('Number of files = ', len(img_test))
        path_size=([input_shape[0],input_shape[1]])

        for i in range(len(img_test)):
            print('Processing files:', i/len(img_test))
            ortho_path=img_test[i]
            predicted_batches=[]
            imageName = os.path.split(ortho_path)[-1]
            outName = imageName.replace('RGB', "CLS")
            test_image=load_img(ortho_path)
            if pre_patch:
                test_image=Img2Patch(test_image,patch_size=path_size,overlap_rati=overlap)
            if pre_augment:
                test_image=PreAug(test_image)

            N = len(test_image) #total number of images
            idx = range(N)
            batchInds = get_batch_inds(batch_size, idx, N,predict=True) 
            for inds in range(len(batchInds)):
                img_batch = [test_image[ind] for ind in batchInds[inds]]
                img_batch=np.array(img_batch,np.float32)
                pred = model.predict(img_batch)
                pred = np.argmax(pred, axis=-1).astype('uint8')
                predicted_batches.extend(pred)
            
            if pre_augment:
                preds=PreAugBack(predicted_batches)
            if pre_patch:
                preds=Patch2Img(preds)
            else:
                preds=predicted_batches

            pred=convert_labels(pred,self.config,toLasStandard=True)

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
    weight_file='weights.35.hdf5'
    weight_path=os.path.join(params.CHECKPOINT_DIR,net_name,weight_file)
    results_folder=os.path.join(params.OUTPUT_DIR,net_name+weight_file)
    num_class=params.NUM_CATEGORIES
    ####data_folder,weight_path,pre_augment=False,pre_patch=False, results_folder='',net_name='',num_class=''
    detector.test(data_folder,weight_path,pre_augment=False,pre_patch=False)

if __name__ == '__main__':

    data_folder=r'C:\TianZhi2019\data'
    train_net(data_folder)
    #test_net('C:/TrainData/track3/Test_new/')
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