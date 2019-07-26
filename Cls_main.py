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
from grss_data import GetPredictData,input_generator,input_generator_mp,GetPredictData_new,track3_Merge_temparal_results_new
from grss_models import get_model
from track1_data import load_all_data_files,input_load_train_data,input_generator_online_process,input_generator_RGB
from grss_data import load_all_data_files_balanced_patches,load_all_data_files_balanced_patches_singel

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
   


    def train(self,data_folder,weighted_loss=False,patch_balance=False,pre_trained_weight_path=''):
        net_name='unet_rgb_c'
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


    def test_track3(self,data_folder,net_name,weight_path,results_folder,num_class=5,only_submit_region=False):
        if os.path.exists(results_folder)==0:
            os.mkdir(results_folder)
        
        from track1_data import load_all_data_test,load_all_data_test_val
        from DataAugment import goback_auger_p,patch2img,GetPatchWeight,patch2img_height
        from dataFunctions import convert_labels
        from track3_data import get_batch_inds
        #####
        channels=net_name.split('_')[1]
        #img_test=load_all_data_test_val(data_folder,channels)
        img_test=load_all_data_test(data_folder,channels)
        #        img_test=img_test[119:301]
        #        dsm_test=dsm_test[119:301]
        ###########load model
        input_shape=[1024,1024,3]

        model,input_shape=get_model(net_name,num_class,weight_path,input_shape=input_shape)
        extra_dsm=False
        extra_class=False
        convertLab=False
        if net_name[-1:]=='c':
            task='CLS'
        else:
            task='AGL'
        if net_name=='unet_rgbh_c' or net_name=='unet_rgbh_h':
            extra_dsm=True
        elif net_name=='unet_rgbc_h' or net_name=='unet_rgbc_c':
            extra_class=True
        patch_weights=GetPatchWeight([input_shape[0],input_shape[1]],pad=64,last_value=0.05)
        batch_size=1
        overlap=0.00
        resize2size=()
            
        print('Number of files = ', len(img_test))
        path_size=([input_shape[0],input_shape[1]])

        for i in range(len(img_test)):
            print('Processing files:', i/len(img_test))
            ortho_path=img_test[i]
            if 'RIGHT' in ortho_path:
                continue
            predicted_patches=[]
            imageName = os.path.split(ortho_path)[-1]
            outName = imageName.replace('RGB', task)
            extra_path=''
            if extra_dsm:
                extra_path=os.path.join(data_folder,'pred_dsm',imageName.replace('RGB', 'AGL')) ## DSP for track 2, other AGL
                convertLab=False
            if extra_class:
                extra_path=os.path.join(data_folder,'pred_class',imageName.replace('RGB', 'CLS'))
                convertLab=True
            #convertLab=False
            img_augs,img_size_o=GetPredictData(ortho_path,channels=channels,extra_path=extra_path,path_size=path_size,convertLab=convertLab,overlap_ratio=overlap,resize2size=resize2size)#(ortho_path, dsm_path='',path_size=(256,256),overlap_ratio=0.5,convertLab=False,normalize_dsm=0,img_ioa=[]):
            if 0:
                mm=0
                for imm in img_augs:
                    imm=imm.transpose(2,0,1)
                    outName_s=outName[:-4]+str(mm)+'-.tif'
                    mm=mm+1
                    tifffile.imsave(os.path.join(results_folder,outName_s),imm)
            N = len(img_augs) #total number of images

            idx = range(N)

            batchInds = get_batch_inds(batch_size, idx, N,predict=True) 
            mm=0
            for inds in range(len(batchInds)):
                img_batch = [img_augs[ind] for ind in batchInds[inds]]
                img_batch=np.array(img_batch,np.float32)
                
                if 0:
                    plt.imshow(img_batch[0,:,:,:])
                    plt.show()
                pred = model.predict(img_batch)
             #  if self.mode == self.config.SEMANTIC_MODE:
                if task=='CLS':
                    pred = np.argmax(pred, axis=-1).astype('uint8')
                    if 0:
                        for pr in pred:
                           # pr=pred
                            outName_s=outName[:-4]+str(mm)+'_888.tif'
                            mm=mm+1
                            tifffile.imsave(os.path.join(results_folder, outName_s), pr, compress=6)

                predicted_patches.extend(pred)
            # preds=goback_auger_p(predicted_patches)
            preds=predicted_patches
            if 0:
                mm=0
                for imm in preds:
                    outName_s=outName[:-4]+str(mm)+'l-.tif'
                    mm=mm+1
                    tifffile.imsave(os.path.join(results_folder,outName_s),imm,compress=6)
            if task=='CLS':
                #pred=patch2img(preds,img_size_o,patch_weights,num_class=num_class, overlap=overlap)
                pred=convert_labels(pred,self.config,toLasStandard=True)
            else:
                pred=patch2img_height(preds,img_size_o,patch_weights,num_class=num_class, overlap=overlap)
                need_move=1
                if need_move:
                    #pred=(pred+1)*12
                    pred=pred+10
           # pred=convert_labels(pred,config,toLasStandard=True)
            #whole_img=(pred*50).astype('uint8')
            tifffile.imsave(os.path.join(results_folder, outName), pred, compress=6)
            #tifffile.imsave(os.path.join(results_folder, outName[:-4]+str(mm)+'l-.tif'), whole_img, compress=6)
    
    def test_track3_new(self,data_folder,net_name,weight_path,results_folder,num_class=5,only_submit_region=False):
        if os.path.exists(results_folder)==0:
            os.mkdir(results_folder)
        from DataAugment import img2patches
        from track1_data import load_all_data_test
        from DataAugment import goback_auger_p,patch2img,GetPatchWeight,patch2img_height,patch2img_new
        from dataFunctions import convert_labels
        from track3_data import get_batch_inds
        #####
        channels=net_name.split('_')[1]
        img_test=load_all_data_test(data_folder,channels)
       # img_test=load_all_data_test_val(data_folder,channels)
        #        img_test=img_test[119:301]
        #        dsm_test=dsm_test[119:301]
        ###########load model
        input_shape=[512,512,3]

        model,input_shape=get_model(net_name,num_class,weight_path,input_shape=input_shape)
        extra_dsm=False
        extra_class=False
        convertLab=False
        if net_name[-1:]=='c':
            task='CLS'
        else:
            task='AGL'
        if len(channels)==4 and channels[-1]=='h':
            extra_dsm=True
        elif len(channels)==4 and channels[-1]=='c':
            extra_class=True
        patch_weights=GetPatchWeight([input_shape[0],input_shape[1]],pad=64,last_value=0.05)
        batch_size=10
        overlap=0.80
        resize2size=()
            
        print('Number of files = ', len(img_test))
        path_size=([input_shape[0],input_shape[1]])

        for i in range(len(img_test)):
            print('Processing files:', i/len(img_test))
            if 'LEFT' in img_test[i]:
                continue
            ortho_path=img_test[i]
            predicted_patches=[]
            imageName = os.path.split(ortho_path)[-1]
            outName = imageName.replace('RGB', task)
            extra_path=''
            if extra_dsm:
                extra_path=os.path.join(data_folder,'pred_dsm',imageName.replace('RGB', 'DSP'))
                convertLab=False
            if extra_class:
                extra_path=os.path.join(data_folder,'pred_class',imageName.replace('RGB', 'CLS'))
                convertLab=True
            #convertLab=False
            #img_augs,img_size_o=GetPredictData(ortho_path,channels=channels,extra_path=extra_path,path_size=path_size,convertLab=convertLab,overlap_ratio=overlap,resize2size=resize2size)#(ortho_path, dsm_path='',path_size=(256,256),overlap_ratio=0.5,convertLab=False,normalize_dsm=0,img_ioa=[]):
            img_augs,img_size_o=GetPredictData_new(ortho_path,channels=channels,extra_path=extra_path,path_size=path_size,convertLab=convertLab,overlap_ratio=overlap,resize2size=resize2size)
            mm=0
            for img in img_augs:
                predicted_patches=[]
                pathces=img2patches(img,patch_size=path_size,overlap_rati=overlap)
                if 0:
                    plt.subplot(131) #用于显示多个子图121代表行、列、位置
                    plt.imshow(img)
                    plt.title('org')
                    plt.subplot(132)
                    plt.imshow(pathces[0])
                    plt.title('rote90') #添加标题
                    plt.subplot(133)
                    plt.imshow(pathces[1])
                    plt.title('rote90_2') #添加标题
                    plt.show()
                N = len(pathces)
                idx = range(N)
                batchInds = get_batch_inds(batch_size, idx, N,predict=True) 
                for inds in range(len(batchInds)):
                    img_batch = [pathces[ind] for ind in batchInds[inds]]
                    img_batch=np.array(img_batch,np.float32)
                    pred_o = model.predict(img_batch)
                    #xx=softmax(pred_o,-1)
                    # xx=np.exp(pred_o)
                    # mm=np.sum(np.exp(pred_o),axis=-1)


                    if task=='CLS':
                        #pred= np.exp(pred_o)/np.sum(np.exp(pred_o),axis=-1,keepdims=True)
                        pred = np.argmax(pred_o, axis=-1).astype('uint8')
                    if 0:
                        plt.subplot(131) #用于显示多个子图121代表行、列、位置
                        plt.imshow(img_batch[0])
                        plt.title('org')
                        plt.subplot(132)
                        plt.imshow(pred[0])
                        plt.title('rote90') #添加标题
                        plt.subplot(133)
                        plt.imshow(pred[1])
                        plt.title('rote90_2') #添加标题
                        plt.show()
                    predicted_patches.extend(pred)
                #preds=goback_auger_p(predicted_patches)

                if task=='CLS':
                    pred=patch2img_new(predicted_patches,img_size_o,patch_weights,num_class=num_class, overlap=overlap)
                    pred=np.rot90(pred,mm)
        
                    #pred=convert_labels(pred,config,toLasStandard=True)
                else:
                    pred=patch2img_height(predicted_patches,img_size_o,patch_weights,num_class=num_class, overlap=overlap)
                    pred=np.rot90(pred,mm)
           # pred=convert_labels(pred,config,toLasStandard=True)
               # whole_img=(pred*50).astype('uint8')
                tifffile.imsave(os.path.join(results_folder, outName[:-4]+str(mm)+'.tif'), pred, compress=6)
               # tifffile.imsave(os.path.join(results_folder, outName[:-4]+str(mm)+'l-.tif'), whole_img, compress=6)
                mm=mm+1
        #resultFolder=results_folder
        #out_folder='../data/validate/track1-rgb_c-morenew-new20190321-merge1/'
        #track3_Merge_temparal_results_new(resultFolder, out_folder,track='track2')




def train_net(data_folder):
    detector=Class_net(params)
    pre_trained_weight_path=''
    weighted_loss=True
    patch_balance=False
    detector.train(data_folder=data_folder,weighted_loss=weighted_loss,patch_balance=patch_balance)

def test_net(data_folder,is_merge=False):
    from track1_data import Merge_all_results_with_baseline,Merge_temparal_results
    detector=Class_net(params,0)
    #weight_path=os.path.join('./track1-rgbc_h-gan-moresamples/', 'gen_weights_epoch8.h5')
    # results_folder='../data/validate/Track3_bridge_psp_1-/'
    weight_path=os.path.join('./track1-step2/','weights.10.hdf5')
    results_folder='../data/re_train/strategy1'
    num_class=params.NUM_CATEGORIES
    #num_class=3
    net_name='unet_rgb_c'
    #net_name='unet_rgb_h'
    #net_name='unet_rgb_c'
    #net_name='unet_msi_c'
 #   class_id=4  data_folder,net_name,weight_path,results_folder,num_class=params.NUM_CATEGORIES,only_submit_region=False):
    #detector.test_track3_new(data_folder,net_name,weight_path,results_folder,num_class=num_class)
    detector.test_track3(data_folder,net_name,weight_path,results_folder,num_class=num_class)
    # if 1:
    #        return
    if net_name[-1]=='c' and is_merge:
        baseline='../data/validate/Track1-test-submit'
        rgb_c_folder='../data/validate/Track1-rgb_c-20190318e30_stratege22--'#D:/grss2019/data/validate/Track1-rgb_c-20190311e27-new70000'
        rgbh_folder='../data/validate/Track1-rgbh_c-20190318e24_stratege2'#D:/grss2019/data/validate/Track1-rgbh_c-gan20190311e21-new70000-newdsm'
        all_result_folders={}
        all_result_folders[0]=rgb_c_folder
        all_result_folders[1]=rgbh_folder
        all_result_folders[2]=baseline
        merged_folder='../data/validate/Track1_class_merge_190318_t-final'
    
        Merge_all_results_with_baseline(all_result_folders,merged_folder)
    else:
        if net_name[-1]=='h':
            from grss_data import constrain_height
            label_folder='C:/TrainData/Track1/Test-Track1/pred_class'
            dsmout_folder='../data/validate/Track1-rgbc_h-gan20190314e11-constrain/'
            constrain_height(results_folder,label_folder,dsmout_folder)
if __name__ == '__main__':

    train_net('C:/TrainData/Track1/train/patch_512')
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