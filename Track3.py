__author__ = 'jhuapl'
__version__ = 0.1

import numpy as np

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
import params

from track3_data import load_all_data_files,input_generator_RGBH, weighted_categorical_crossentropy,get_batch_inds
from dataFunctions import my_weighted_loss_5_classes,my_weighted_loss
class Track3_net:
    def __init__(self, params=None, mode=None):
        """
        Initializes baseline by setting params, mode, and loading mean values to be used in the case non-RGB imagery is being used
        :param params: input parameters from params.py
        :param mode: either SEMANTIC_MODE or SINGLEVIEW_MODE from params (0 or 1)
        :return: None
        """
        self.params=params
        self.mode = mode
        self.meanVals = None
        if params.NUM_CHANNELS != 3:
            if params.MEAN_VALS_FILE is None:
                self.meanVals = np.zeros(params.NUM_CHANNELS).tolist()
            else:
                self.meanVals = json.load(open(params.MEAN_VALS_FILE))

        # TensorFlow allocates all GPU memory up front by default, so turn that off
        config = tf.ConfigProto()
        os.environ["CUDA_VISIBLE_DEVICES"]=params.GPUS
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

    def test(self):
        """
        Launches testing, which saves output files
        :return: None
        """
        input_tensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS))
        input_shape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS)
        model = self.get_model(None, input_tensor, input_shape)
        
        if self.mode == self.params.SEMANTIC_MODE:
            numPredChannels = self.params.NUM_CATEGORIES
            outReplaceStr = self.params.CLSPRED_FILE_STR
            model = self.build_model()
            model.load_weights(self.params.SEMANTIC_TEST_MODEL, by_name=True)
        elif self.mode == self.params.SINGLEVIEW_MODE:
            numPredChannels = 1
            outReplaceStr = self.params.AGLPRED_FILE_STR
            model = self.build_model()
            model.load_weights(self.params.SINGLEVIEW_TEST_MODEL, by_name=True)
        
        model.summary()

        imgPaths = get_image_paths(self.params, isTest=True)
        print('Number of files = ', len(imgPaths))
        for imgPath in tqdm(imgPaths):
            imageName = os.path.split(imgPath)[-1]
            outName = imageName.replace(self.params.IMG_FILE_STR, outReplaceStr)
            img=load_img(imgPath)
            if img.shape[1]!=self.params.IMG_SZ[0] or img.shape[2]!=self.params.IMG_SZ[1]:
                img=cv2.resize(img,(self.params.IMG_SZ[0],self.params.IMG_SZ[1]))

            img = np.expand_dims(img, axis=0).astype('float32')
            img = image_batch_preprocess(img, self.params, self.meanVals)

            
            pred = model.predict(img)[0,:,:,:]
            
            if self.mode == self.params.SEMANTIC_MODE:
                if self.params.NUM_CATEGORIES > 1:
                    pred = np.argmax(pred, axis=2).astype('uint8')
                else:
                    pred = (pred > self.params.BINARY_CONF_TH).astype('uint8')
                if self.params.CONVERT_LABELS:
                    pred = convert_labels(pred, self.params, toLasStandard=True)
            else:
                pred = pred[:,:,0]
                
            tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred, compress=6)
            
    def test_patch(self):
        """
        Launches testing, which saves output files
        :return: None
        """
        from DataAugment import predict_data_generator,goback_auger_p,patch2img
        input_tensor = Input(shape=(self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS))
        input_shape = (self.params.IMG_SZ[0],self.params.IMG_SZ[1],self.params.NUM_CHANNELS)
        model = self.get_model(None, input_tensor, input_shape)
        
        if self.mode == self.params.SEMANTIC_MODE:
            numPredChannels = self.params.NUM_CATEGORIES
            outReplaceStr = self.params.CLSPRED_FILE_STR
            model = self.build_model()
            model.load_weights(self.params.SEMANTIC_TEST_MODEL, by_name=True)
        elif self.mode == self.params.SINGLEVIEW_MODE:
            numPredChannels = 1
            outReplaceStr = self.params.AGLPRED_FILE_STR
            model = self.build_model()
            model.load_weights(self.params.SINGLEVIEW_TEST_MODEL, by_name=True)
        
        model.summary()

        imgPaths = get_image_paths(self.params, isTest=True)
        print('Number of files = ', len(imgPaths))
        for imgPath in tqdm(imgPaths):
            imageName = os.path.split(imgPath)[-1]
            outName = imageName.replace(self.params.IMG_FILE_STR, outReplaceStr)
            img_augs=predict_data_generator(imgPath)
            if 0:
                mm=0
                for imm in img_augs:
                    outName_s=outName[:-4]+str(mm)+'-.tif'
                    mm=mm+1
                    tifffile.imsave(os.path.join(self.params.OUTPUT_DIR,outName_s),imm,compress=6)
         #   img=load_img(imgPath)
            predicted_patches=[]
 
            for patch in img_augs:
                #img=np.array(patch,np.float32)

                img = np.expand_dims(patch, axis=0).astype('float32')
                img = image_batch_preprocess(img, self.params, self.meanVals)

            
                pred = model.predict(img)
            
          #  if self.mode == self.params.SEMANTIC_MODE:
                if self.params.NUM_CATEGORIES > 1:
                    pred = np.argmax(pred, axis=-1).astype('uint8')
                    if 0:
                        count=0
                        outName_s=outName[:-4]+str(count)+'.tif'
                        count=count+1
                        tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName_s), pred, compress=6)
                else:
                    pred = (pred > self.params.BINARY_CONF_TH).astype('uint8')
                if self.params.CONVERT_LABELS:
                    pred = convert_labels(pred, self.params, toLasStandard=True)
                predicted_patches.append(pred)
            preds=goback_auger_p(predicted_patches)

            if 0:
                mm=0
                for imm in preds:
                    outName_s=outName[:-4]+str(mm)+'l-.tif'
                    mm=mm+1
                    tifffile.imsave(os.path.join(self.params.OUTPUT_DIR,outName_s),imm,compress=6)

            img_size_o=(1024,1024)
            overlap=0.5
            pred=patch2img(preds,img_size_o,overlap)
            tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred, compress=6)
            

    def train_bb(self):
        """
        Launches training and stores checkpoints at a frequency defined within params
        :return: None
        """
        image_paths =get_image_paths_bb(self.params, isTest=False) ## for original whole image
        #image_paths = get_image_paths(self.params, isTest=False) ##for patch
        if self.params.BLOCK_IMAGES:
            train_data = get_train_data(image_paths, self.params)
        else:
            train_data = []
            for imgPath in image_paths:
                train_data.append((imgPath, 0, 0))
        #img_data, label_data=load_all_data(imgs,labels,self.params, self.mode)    
        train_datagen = self.image_generator(train_data)
        model = self.build_model()
        model.summary()
        csv_logger = CSVLogger(os.path.join(self.params.CHECKPOINT_DIR,'train.csv'))

        checkpoint = ModelCheckpoint(filepath=self.params.CHECKPOINT_PATH, monitor='loss', verbose=0, save_best_only=False,
                                         save_weights_only=False, mode='auto', period=self.params.MODEL_SAVE_PERIOD)
        callbacks=[csv_logger,  checkpoint]
        if len(train_data) <= 0:
            raise ValueError("No training data found. Update params.py accordingly")

        # train model 
        csv_logger
        model.fit_generator(train_datagen, steps_per_epoch=int(len(train_data)/self.params.BATCH_SZ),
                            epochs=self.params.NUM_EPOCHS, callbacks=callbacks)
    def train_track3_single(self,net_name='unet',check_folder=params.CHECKPOINT_DIR,pre_trained_weight_path=''):
        if os.path.exists(check_folder)==0:
            os.mkdir(check_folder)
        CHECKPOINT_PATH = os.path.join(check_folder, 'weights.{epoch:02d}.hdf5')
        from track3_data import load_all_data_files_train_single_class,input_load_train_data,weighted_categorical_crossentropy,input_generator_RGBH
        text_files_positive='C:/TrainData/Track3/Train/patch_256/bridge_list.txt'
        text_files_negative='C:/TrainData/Track3/Train/patch_256/bridge_list_no.txt'
#        text_files_positive='C:/TrainData/Track3/Train/patch_512/tree_list.txt'
#        text_files_negative='C:/TrainData/Track3/Train/patch_512/tree_list_no.txt'
        Dataset_path='C:/TrainData/Track3/Train/patch_256/'
        img_file_t, dsm_file_t, lable_file_t, img_file_v, dsm_file_v, label_file_v=load_all_data_files_train_single_class(text_files_positive,text_files_negative,Dataset_path)
        single_class_id=4
        nb_epoch=200
        batch_size=22
        n_batch_per_epoch=len(img_file_t)//batch_size

        num_val_sample=len(img_file_v)
        n_batch_per_epoch_val=num_val_sample//batch_size
        data_train, label_train=input_load_train_data(img_file_t,dsm_file_t, lable_file_t,single_class_id)
        data_val, label_val=input_load_train_data(img_file_v,dsm_file_v, label_file_v,single_class_id)
        #train_generator=input_generator_RGBH(img_file_t,dsm_file_t, lable_file_t, batch_size)
        #val_generator = input_generator_RGBH(img_file_v,dsm_file_v,label_file_v,batch_size)
##########load model
        single_class=1
        if single_class:
            number_class=2
            class_weight=[1,50]
        else:
            number_class=params.NUM_CATEGORIES
            class_weight=[1.0,10.0,10.0,20.,30.]
        if net_name=='psp':
            input_shape=(473,473,9)
        elif net_name=='unet':
            input_shape = [256,256,9]
        else:
            print("the input net is not exist")
            return
        weight_path=pre_trained_weight_path
        model=self.get_model(net_name,input_shape,number_class,class_weight=class_weight,weight_path=weight_path)

#################
        # NUM_CATEGORIES=2
        
        # if net=='psp':
        #     model_name='pspnet101_cityscapes'
        #     input_shape=(473,473,9)
        #     model = pspnet.PSPNet101(nb_classes=NUM_CATEGORIES, input_shape=input_shape,
        #                                weights=model_name)
        #     model=model.model
        # elif net=='unet':
        #     input_shape = [256,256,9]
        #     from keras.layers import Input
        #     input_tensor = Input(shape=(input_shape[0],input_shape[1],input_shape[2]))
        #     model = Unet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=params.BACKBONE, 
        #                         encoder_weights=None, classes=2)
                                
        #     model.load_weights(os.path.join('./checkpoint_track3-1/', 'weights.80.hdf5'))
        #     #loss=params.SEMANTIC_LOSS
        # #loss=my_weighted_loss
        # from keras.optimizers import Adam, SGD
        # from keras.callbacks import ModelCheckpoint,CSVLogger

        # optimizer=Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # weights = np.array([1.0,20.0])
        # loss = weighted_categorical_crossentropy(weights)
        # model.compile(optimizer, loss=loss)
        # model.summary()
        csv_logger = CSVLogger(os.path.join(check_folder,'train.csv'))

        checkpoint = ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor='loss', verbose=1, save_best_only=False,
                                        save_weights_only=True, mode='auto', period=params.MODEL_SAVE_PERIOD)
        callbacks=[csv_logger,  checkpoint]

#        model.fit_generator(train_generator, steps_per_epoch=n_batch_per_epoch,
#                            validation_data=val_generator, validation_steps=n_batch_per_epoch_val,
#                            epochs=nb_epoch, callbacks=callbacks)

        model.fit(np.array(data_train),np.array(label_train), batch_size=batch_size,validation_data=(np.array(data_val), np.array(label_val)),
                             epochs=nb_epoch, callbacks=callbacks)

    def Merge_track3_results(self, result_folder,out_folder,offset_folder,num_class=params.NUM_CATEGORIES):
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
        NUM_CATEGORIES=num_class
        for m in range(len(site_names)):
            imgs=site_images[m]
            im=cv2.imread(imgs[0],0)
            vote_map=np.zeros((im.shape[0],im.shape[1],NUM_CATEGORIES))
            for img_p in imgs:
                im=cv2.imread(img_p,0)
                one_hot=to_categorical(im,NUM_CATEGORIES)
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
 
    def get_model_bb(self, encoderWeights, input_tensor, input_shape):
        """
        Loads the model from segmentation_models for either semantic segmentation or single-view depth prediction
        :param encoderWeights: encoder weights for the backbone (e.g., imagenet)
        :param input_tensor: image shape for training
        :param input_shape: tensor shape for training
        :return model to be used for training/testing:
        """
        
        if self.mode == self.params.SINGLEVIEW_MODE:
            model = UnetRegressor(input_shape=input_shape, input_tensor=input_tensor, 
                        backbone_name=self.params.BACKBONE, encoder_weights=encoderWeights)
        elif self.mode == self.params.SEMANTIC_MODE:
            model = Unet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=self.params.BACKBONE, 
                         encoder_weights=encoderWeights, classes=self.params.NUM_CATEGORIES)
        # elif self.mode == self.params.SEMANTIC_MODE:
        #     model = PSPNet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=self.params.BACKBONE, 
        #                  encoder_weights=encoderWeights, classes=self.params.NUM_CATEGORIES)
        
        # elif self.mode == self.params.SEMANTIC_MODE:
        #     model_name='pspnet101_cityscapes'
        #     input_shape=(713,713)
        #     pspnet_model = pspnet.PSPNet101(nb_classes=self.params.NUM_CATEGORIES, input_shape=input_shape,
        #                                weights=model_name)
        #     model=pspnet_model.model
            # model = pspnet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=self.params.BACKBONE, 
            #               encoder_weights=encoderWeights, classes=self.params.NUM_CATEGORIES)
        
        return model
    
    def build_model(self):
        """
        Prepares either the single-view depth prediction model or the semantic segmentation model
            depending on params.py
        :return: model architecture to train
        """
               
        if self.params.BLOCK_IMAGES:
            imgSz = self.params.BLOCK_SZ
        else:
            imgSz = self.params.IMG_SZ
        
        input_tensor = Input(shape=(imgSz[0],imgSz[1],self.params.NUM_CHANNELS))
        input_shape = (imgSz[0],imgSz[1],self.params.NUM_CHANNELS)
        
        if self.params.CONTINUE_TRAINING:
            model = self.get_model(None, input_tensor, input_shape)
            if self.mode == self.params.SINGLEVIEW_MODE:
                print('Continuing trainining from %s' % self.params.CONTINUE_SINGLEVIEW_MODEL_FILE)
                model.load_weights(self.params.CONTINUE_SINGLEVIEW_MODEL_FILE)
            elif self.mode == self.params.SEMANTIC_MODE:
                print('Continuing trainining from %s' % self.params.CONTINUE_SEMANTIC_MODEL_FILE)
                model.load_weights(self.params.CONTINUE_SEMANTIC_MODEL_FILE)
        else:
            if self.params.NUM_CHANNELS > 3:
                input_tensor_rgb = Input(shape=(imgSz[0],imgSz[1],3))
                input_shape_rgb = (imgSz[0],imgSz[1],3)
                model = self.get_model(None, input_tensor, input_shape)
                if self.params.ENCODER_WEIGHTS is not None:
                    baseModel = self.get_model(self.params.ENCODER_WEIGHTS, input_tensor_rgb, input_shape_rgb)
                    print("Copying %s weights to %d-band model" % (self.params.ENCODER_WEIGHTS, self.params.NUM_CHANNELS))
                    for i in tqdm(range(len(baseModel.layers))):
                        if i>=7:
                            model.layers[i].set_weights(baseModel.layers[i].get_weights())
            else:
                model = self.get_model(self.params.ENCODER_WEIGHTS, input_tensor, input_shape)
            
        #loss = self.params.SEMANTIC_LOSS
        loss=my_weighted_loss
        if self.mode == self.params.SINGLEVIEW_MODE:
            loss=self.no_nan_mse
        optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer, loss=loss)
        
        return model
    
    def image_generator(self, trainData):
        """
        Generates training batches of image data and ground truth from either semantic or depth files
        :param trainData: training paths of CLS files (string replacement to get RGB and depth files) and starting x,y pixel positions (can be      non-zero if blocking is set to happen in params.py)
        :yield: current batch data
        """
        idx = np.random.permutation(len(trainData))
        while True:
            batchInds = get_batch_inds(idx, self.params)
            for inds in batchInds:
                imgBatch,labelBatch = load_batch(inds, trainData, self.params, self.mode, self.meanVals)  ##for patch
               # imgBatch,labelBatch =load_batch_bb(inds, trainData, self.params, self.mode, self.meanVals)  ## for original image
                yield (imgBatch, labelBatch)
                #return (imgBatch, labelBatch)

    def no_nan_mse(self, y_true, y_pred):
        """
        Custom mean squared error loss function for single-view depth prediction used to ignore NaN/invalid depth values
        :param y_true: ground truth depth
        :param y_pred: predicted depth
        :return: mse loss without nan influence
        """
        mask_true = K.cast(K.not_equal(y_true, self.params.IGNORE_VALUE), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
        return masked_mse
    def train_track3(self,data_folder,net_name='unet',check_folder=params.CHECKPOINT_DIR,pre_trained_weight_path='',single_class=False):
        if os.path.exists(check_folder)==0:
            os.mkdir(check_folder)
        CHECKPOINT_DIR=check_folder
        CHECKPOINT_PATH = os.path.join(check_folder, 'weights.{epoch:02d}.hdf5')
        
#        data_folder='C:/TrainData/Track3/Train/patch_473/'
        img_train, dsm_train, lable_train, img_val, dsm_val, label_val=load_all_data_files(data_folder)
        
        num_training_sample=len(img_train)
        batch_size=22
        n_batch_per_epoch=num_training_sample//batch_size

        num_val_sample=len(img_val)
        n_batch_per_epoch_val=num_val_sample//batch_size

        nb_epoch=300

        train_generator=input_generator_RGBH(img_train,dsm_train, lable_train, batch_size)
        val_generator = input_generator_RGBH(img_val,dsm_val,label_val,batch_size)
##########load model
        if single_class:
            number_class=2
            class_weight=[1,50]
        else:
            number_class=params.NUM_CATEGORIES
            class_weight=[1.0,10.0,10.0,20.,30.]
        if net_name=='psp':
            input_shape=(473,473,9)
        elif net_name=='unet':
            input_shape = [256,256,9]
        else:
            print("the input net is not exist")
            return
        weight_path=pre_trained_weight_path
        model=self.get_model(net_name,input_shape,number_class,class_weight=class_weight,weight_path=weight_path)
        from keras.callbacks import ModelCheckpoint,CSVLogger

        csv_logger = CSVLogger(os.path.join(CHECKPOINT_DIR,'train.csv'))

        checkpoint = ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor='loss', verbose=1, save_best_only=False,
                                        save_weights_only=True, mode='auto', period=params.MODEL_SAVE_PERIOD)
        callbacks=[csv_logger,  checkpoint]

        model.fit_generator(train_generator, steps_per_epoch=n_batch_per_epoch,
                            validation_data=val_generator, validation_steps=n_batch_per_epoch_val,
                            epochs=nb_epoch, callbacks=callbacks)

    def get_model(self,net_name,input_shape,number_class,class_weight,weight_path):
        from segmentation_models import pspnet#PSPNet
        if net_name=='psp':
            model_name='pspnet101_cityscapes'
          #  input_shape=(473,473,9)
            model = pspnet.PSPNet101(nb_classes=number_class, input_shape=input_shape,
                                        weights=model_name)
            model=model.model
        if net_name=='psp_50':
          #  input_shape=(473,473,9)
            model_name='pspnet50_ade20k'
            #output_mode='sigmoid'
            pspnet = pspnet.PSPNet101(nb_classes=number_class, input_shape=input_shape,
                                        weights=model_name)
            model=model.model
        elif net_name=='unet':
           # input_shape = [256,256,9]
            from keras.layers import Input
            input_tensor = Input(shape=(input_shape[0],input_shape[1],input_shape[2]))
            model = Unet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=params.BACKBONE, 
                                encoder_weights=None, classes=number_class)
        ##[1.0,10.0,10.0,20.,30.]
        weights = np.array(class_weight)
#        loss = weighted_categorical_crossentropy(weights)
        if number_class==2:
            loss=my_weighted_loss
        elif number_class==5:
            loss=my_weighted_loss_5_classes
#        loss=params.SEMANTIC_LOSS
        optimizer=Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        if (len(weight_path)>2):
            model.load_weights(weight_path)
        model.compile(optimizer, loss=loss)

        model.summary()
        return model        
    def test_track3(self,data_folder,net_name,weight_path,results_folder,single_class=False,only_submit_region=False):
        if os.path.exists(results_folder)==0:
            os.mkdir(results_folder)
        os.environ["CUDA_VISIBLE_DEVICES"]=params.GPUS
        from track3_data import load_all_data_test,GetPredictData
        from DataAugment import goback_auger_p,patch2img,GetPatchWeight
        from dataFunctions import convert_labels
        ortho_txt=os.path.join(data_folder,'test_orthos.txt')
        if only_submit_region:
            dsm_txt=os.path.join(data_folder,'test_dsm_submit.txt')
        else:
            dsm_txt=os.path.join(data_folder,'test_dsm.txt')
        img_test, dsm_test=load_all_data_test(ortho_txt,dsm_txt)
#        img_test=img_test[119:301]
#        dsm_test=dsm_test[119:301]
        
    ###########load model
        if single_class:
            number_class=2
            class_weight=[1,20]
        else:
            number_class=params.NUM_CATEGORIES
            class_weight=[1.0,10.0,10.0,20.,30.]
        if net_name=='psp':
            input_shape=(473,473,9)
        elif net_name=='unet':
            input_shape = [256,256,9]
        else:
            print("the input net is not exist")
            return
        model=self.get_model(net_name,input_shape,number_class,class_weight,weight_path)
        # if net_name=='psp':
        #     model_name='pspnet101_cityscapes'
        #     input_shape=(473,473,9)
        #     model = pspnet.PSPNet101(nb_classes=number_class, input_shape=input_shape,
        #                                weights=model_name)
        #     model=model.model

        # elif net_name=='unet':
        #     input_shape = [256,256,9]
        #     from keras.layers import Input
        #     input_tensor = Input(shape=(input_shape[0],input_shape[1],input_shape[2]))
        #     model = Unet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=params.BACKBONE, 
        #                         encoder_weights=None, classes=number_class)
 

        # loss=params.SEMANTIC_LOSS
        # optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # model.compile(optimizer, loss=loss)
        # model.load_weights(weight_path)
        # model.summary()
    #####
        patch_weights=GetPatchWeight([input_shape[0],input_shape[1]])
        batch_size=40
        overlap=0.66
            
        print('Number of files = ', len(img_test))
        path_size=([input_shape[0],input_shape[1]])
        for i in range(len(img_test)):
            print('Processing files:', i/len(img_test))
            ortho_path = os.path.join(data_folder,'Ortho_result',img_test[i])
            dsm_path=os.path.join(data_folder,'DSM',dsm_test[i])
            predicted_patches=[]
            imageName = os.path.split(ortho_path)[-1]
            outName = imageName.replace('ortho', 'CLS')
            img_ioa=[]
            if only_submit_region:
                offset_file=os.path.join(data_folder,'offset',imageName[:7]+'_DSM.txt')
                offset = np.loadtxt(offset_file)
                offset=offset.astype('int')
                img_ioa=[offset[1],offset[1]+512,offset[0],offset[0]+512]

            img_augs,img_size_o=GetPredictData(ortho_path,dsm_path,path_size,overlap_ratio=overlap,img_ioa=img_ioa)
            if 0:
                
                
                mm=0
                for imm in img_augs:
                    imm=imm.transpose(2,0,1)
                    outName_s=outName[:-4]+str(mm)+'-.tif'
                    mm=mm+1
                    gdal_reader.write_img(os.path.join(results_folder,outName_s),[],[],imm)
                    #tifffile.imsave(os.path.join(params.OUTPUT_DIR_TRACK3,outName_s),imm[:,:,0:3],compress=6)
            #   img=load_img(imgPath)
            
            
            N = len(img_augs) #total number of images

            idx = range(N)

            batchInds = get_batch_inds(batch_size, idx, N,predict=True) 
            mm=0
            for inds in range(len(batchInds)):
                img_batch = [img_augs[ind] for ind in batchInds[inds]]
                img_batch=np.array(img_batch,np.float32)
                img_batch=np.squeeze(img_batch)
                #patch=img_augs[get_batch_inds]
                img_batch[:,:,:,0:-1]=img_batch[:,:,:,0:-1]/256.0-1
                img_batch[:,:,:,-1]=img_batch[:,:,:,-1]/50.0-1
                #img = np.expand_dims(img_batch, axis=0).astype('float32')
                pred = model.predict(img_batch)
    ########    
            # for patch in img_augs:
            #     #img=np.array(patch,np.float32)
            #     patch[:,:,0:-1]=patch[:,:,0:-1]/256.0-1
            #     patch[:,:,-1]=patch[:,:,-1]/50.0-1
            #     img = np.expand_dims(patch, axis=0).astype('float32')
            #     pred = model.predict(img)
    ##################        
            #  if self.mode == self.params.SEMANTIC_MODE:
                if params.NUM_CATEGORIES > 1:
                    pred = np.argmax(pred, axis=-1).astype('uint8')
                    if 0:
                     #   for pr in pred:
                            pr=pred
                            outName_s=outName[:-4]+str(mm)+'_888.tif'
                            mm=mm+1
                            tifffile.imsave(os.path.join(results_folder, outName_s), pr, compress=6)
                else:
                    pred = (pred > params.BINARY_CONF_TH).astype('uint8')
                if params.CONVERT_LABELS:
                    pred = convert_labels(pred, params, toLasStandard=True)
                #pred=pred.tolist()
                predicted_patches.extend(pred)
            preds=goback_auger_p(predicted_patches)

            if 0:
                mm=0
                for imm in preds:
                    outName_s=outName[:-4]+str(mm)+'l-.tif'
                    mm=mm+1
                    tifffile.imsave(os.path.join(results_folder,outName_s),imm,compress=6)
           # proj2,geotrans2,dsm=gdal_reader.read_img(dsm_path)
        # dsm=cv2.imread(dsm_path,0)
            #img_size_o=(dsm.shape[0],dsm.shape[1])
            
           # preds=np.array(preds)
            pred=patch2img(preds,img_size_o,patch_weights,num_class=number_class, overlap=overlap)
            tifffile.imsave(os.path.join(results_folder, outName), pred, compress=6)

def test_single_classification():
    detector=Track3_net(params,0)
    weight_path=os.path.join('./checkpoint_track3-single/', 'weights.200.hdf5')
    results_folder='../data/validate/Track3-Submission-bridge/'
    detector.test_track3(weight_path,results_folder,single_class=True,)

def train_net(data_folder):
    detector=Track3_net(params,0)
    net='unet'
    pre_trained_weight_path=os.path.join('./checkpoint_track3-unet_bridge/', 'weights.120.hdf5')
    #pre_trained_weight_path=os.path.join('./checkpoint_track3-unet_all_fixed_classweight_2','weights.80.hdf5')
    check_folder='./checkpoint_track3_unet_bridge_dynamic_classweight_newnormal/'
    single_class=True
    #detector.train_track3_single(net_name=net,check_folder=check_folder,pre_trained_weight_path=pre_trained_weight_path)
    detector.train_track3(net_name=net,data_folder=data_folder,check_folder=check_folder,
                            pre_trained_weight_path=pre_trained_weight_path,single_class=single_class)
def test_net(data_folder):
    from track3_data import Merge_all_results_with_baseline,Merge_temparal_results
    detector=Track3_net(params,0)
    # weight_path=os.path.join('./checkpoint_track3_single_psp_bridge/', 'weights.20.hdf5')
    # results_folder='../data/validate/Track3_bridge_psp_1-/'
    weight_path=os.path.join('./checkpoint_track3_unet_all_dynamic_classweight_newnormal/', 'weights.230.hdf5')
    results_folder='../data/validate/Track3_msi_20190316e230/'
    out_folder='G:/programs/dfc2019-master/track3/data/validate/Track3_msi_orthoMap_20190316e230'
    baseline_folder='G:/programs/dfc2019-master/track3/data/validate/Track3_submission_Merge_dynamic'
    offset_folder='C:/TrainData/Track3/Test_new/offset'
    single_class=False
    only_submit_region=False
    net_name='unet'
    class_id=-1
    detector.test_track3(data_folder,net_name,weight_path,results_folder,only_submit_region=only_submit_region,single_class=single_class,)
#    Merge_temparal_results(results_folder,out_folder,offset_folder,class_id)
    return
    all_result_folders={}
    all_result_folders[0]=out_folder
    all_result_folders[1]=baseline_folder
    #all_result_folders[2]=out_folder
    merged_folder='G:/programs/dfc2019-master/track3/data/validate/Track3_submit20190301_t4'
    
    Merge_all_results_with_baseline(all_result_folders,merged_folder)
if __name__ == '__main__':
 #   test_net('C:/TrainData/Track3/Train')
 #   train_net('C:/TrainData/Track3/Train/patch_256')
    test_net('C:/TrainData/Track3/Test-Track3')

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