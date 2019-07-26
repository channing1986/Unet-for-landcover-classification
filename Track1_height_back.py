__author__ = 'Changlin'
__version__ = 0.1

import numpy as np

from segmentation_models import UnetRegressor,Unet,pspnet#PSPNet
#from segmentation_models import psp_50
from dataFunctions import *
from keras.applications import imagenet_utils
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

from model import models
from track1_data_height import no_nan_mse
from track1_data_height import load_all_data_files,input_load_train_data,input_generator_online_process,input_generator_RGB,load_all_data_files_balanced_patches
from dataFunctions import my_weighted_loss_5_classes,my_weighted_loss
class Track1_net:
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
    
    def train_track3(self,data_folder,net_name='unet',check_folder=params.CHECKPOINT_DIR,pre_trained_weight_path='',single_class=False,use_GAN=False):
        if os.path.exists(check_folder)==0:
            os.mkdir(check_folder)
        CHECKPOINT_DIR=check_folder
        CHECKPOINT_PATH = os.path.join(check_folder, 'weights.{epoch:02d}.hdf5')
        
        val_ratio=0.1
        #img_train,  lable_train, img_val,  label_val=load_all_data_files(data_folder,val_ratio)
        img_train,  dsm_train, lable_train, img_val,dsm_val, label_val=load_all_data_files_balanced_patches(data_folder,val_ratio)
        
        num_training_sample=len(img_train)
        batch_size=6
        n_batch_per_epoch=num_training_sample//batch_size

        num_val_sample=len(img_val)
        n_batch_per_epoch_val=num_val_sample//batch_size

        nb_epoch=300

        train_generator=input_generator_RGB(img_train, dsm_train,lable_train, batch_size)
        val_generator = input_generator_RGB(img_val,dsm_val, label_val,batch_size)
        #########load model
        if single_class:
            number_class=2

        else:
            number_class=params.NUM_CATEGORIES

        if net_name=='psp':
            input_shape=(473,473,3)
        if net_name=='psp_50':
            input_shape=(473,473,3)
        elif net_name=='unet':
            input_shape = [512,512,3]
        elif net_name=='unet_height':
            input_shape = [512,512,3]
            number_class=1
        else:
            print("the input net is not exist")
            return
        weight_path=pre_trained_weight_path
        generator_model=self.get_model(net_name,input_shape,number_class,weight_path=weight_path)

        if use_GAN:
            from keras.utils import generic_utils
            from utils import data_utils
            save_every_epoch=params.MODEL_SAVE_PERIOD
                    # Create optimizers
            opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
              # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
            opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

            from track1_data_height import get_nb_patch
            img_dim=input_shape
            patch_dim=[128,128]
            image_data_format= "channels_last"
            nb_patch, img_dim_disc =get_nb_patch(img_dim, patch_dim, image_data_format)
                    # Load discriminator model
            bn_mode=2
            use_mbd=True
            ##########
            discriminator_model = models.load("DCGAN_discriminator",
                                          img_dim_disc,
                                          nb_patch,
                                          bn_mode,
                                          use_mbd,
                                          batch_size)


            discriminator_model.trainable = False

            DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   img_dim,
                                   patch_dim,
                                   image_data_format)

            loss = [no_nan_mse, 'binary_crossentropy']
            loss_weights =  [1, 5]
            DCGAN_model.compile(loss=loss,loss_weights=loss_weights, optimizer=opt_dcgan)

            discriminator_model.trainable = True
            discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)
        ##############
            gen_loss = 100
            disc_loss = 100

            # Start training
            f= open ("record_4.txt","w") 
            print("Start training")
            for e in range(nb_epoch):
            # Initialize progbar and batch counter
                progbar = generic_utils.Progbar(num_training_sample)
                batch_counter = 1
                start = time.time()

                for x_train, y_train in train_generator:

                # Create a batch to feed the discriminator model
                    X_disc, y_disc = data_utils.get_disc_batch(y_train,
                                                           x_train,
                                                           generator_model,
                                                           batch_counter,
                                                           img_dim_disc,
                                                           image_data_format,
                                                           nb_patch)

                    # Update the discriminator
                    disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)

                    # Create a batch to feed the generator model
                    X_gen,X_gen_target  = next(train_generator)
                    y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                    y_gen[:, 1] = 1

                    # Freeze the discriminator
                    discriminator_model.trainable = False
                    gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                    # Unfreeze the discriminator
                    discriminator_model.trainable = True

                    batch_counter += 1
                    progbar.add(batch_size, values=[("D logloss", disc_loss),
                                                    ("G tot", gen_loss[0]),
                                                    ("G L1", gen_loss[1]),
                                                    ("G logloss", gen_loss[2])])

                    if batch_counter >= n_batch_per_epoch:
                        break
                val_loss= generator_model.evaluate_generator(val_generator, len(img_val) // batch_size)
                f.write(str(disc_loss)+',')
                f.write(str(gen_loss[0])+',')
                #f.write(str(gen_loss[1])+',')
                #f.write(str(gen_loss[2])+',')
                #f.write(str(gen_loss2[0])+',')
                f.write(str(val_loss))
                #f.write(str(gen_loss2[1])+',')
                #f.write(str(gen_loss2[2]))
                f.write('\n')
                print("")
                print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

                if e % save_every_epoch == 0:
                    gen_weights_path = os.path.join(check_folder, 'gen_weights_epoch%s.h5' % ( e))
                    generator_model.save_weights(gen_weights_path, overwrite=True)

                    disc_weights_path = os.path.join(check_folder, 'disc_weights_epoch%s.h5' % ( e))
                    discriminator_model.save_weights(disc_weights_path, overwrite=True)

                    DCGAN_weights_path = os.path.join(check_folder, 'DCGAN_weights_epoch%s.h5' % ( e))
                    DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)
            f.close()
        else:
            from keras.callbacks import ModelCheckpoint,CSVLogger

            csv_logger = CSVLogger(os.path.join(CHECKPOINT_DIR,'train.csv'))

            checkpoint = ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor='loss', verbose=1, save_best_only=False,
                                        save_weights_only=True, mode='auto', period=params.MODEL_SAVE_PERIOD)
            callbacks=[csv_logger,  checkpoint]
            generator_model.fit_generator(train_generator, steps_per_epoch=n_batch_per_epoch,
                            validation_data=val_generator, validation_steps=n_batch_per_epoch_val,
                            epochs=nb_epoch, callbacks=callbacks)


    def get_model(self,net_name,input_shape,number_class,weight_path):
        from segmentation_models import pspnet#PSPNet
        if net_name=='psp':
            model_name='pspnet101_cityscapes'
           # input_shape=(473,473,3)
            model = pspnet.PSPNet101(nb_classes=number_class, input_shape=input_shape,
                                        weights=model_name)
            model=model.model
        elif net_name=='psp_50':
            #input_shape=(473,473,3)
            model_name='pspnet50_ade20k'
            #output_mode='sigmoid'
            model = pspnet.PSPNet50(nb_classes=number_class, input_shape=input_shape,
                                        weights=model_name)
            model=model.model
        elif net_name=='unet':
            #input_shape = [512,512,3]
            from keras.layers import Input
            input_tensor = Input(shape=(input_shape[0],input_shape[1],input_shape[2]))
            model = Unet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=params.BACKBONE, 
                                encoder_weights=None, classes=number_class)
        elif net_name=='unet_height':
                #input_shape = [512,512,3]
            from keras.layers import Input
            input_tensor = Input(shape=(input_shape[0],input_shape[1],input_shape[2]))
            model = UnetRegressor(input_shape=input_shape, input_tensor=input_tensor, 
                        backbone_name=params.BACKBONE)
 
        ##[1.0,10.0,10.0,20.,30.]
        #       loss = weighted_categorical_crossentropy(weights)
        if net_name=='unet_height':
            loss=no_nan_mse
        elif number_class==2:
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
    def test_track3(self,data_folder,net_name,weight_path,results_folder,single_class=False,only_submit_region=False,img_folder_name='imgs'):
        if os.path.exists(results_folder)==0:
            os.mkdir(results_folder)

        from track1_data_height import load_all_data_test,GetPredictData
        from DataAugment import goback_auger_p,patch2img,GetPatchWeight,patch2img_height
        from dataFunctions import convert_labels
        from track3_data import get_batch_inds
        
        img_test=load_all_data_test(data_folder,img_folder_name)
        #     img_test=img_test[119:301]
        #    dsm_test=dsm_test[119:301]
        input_shape_t=[512,512,3]        
            ###########load model
        if single_class:
            number_class=2
        else:
            number_class=params.NUM_CATEGORIES

        if net_name=='psp':
            input_shape=(473,473,3)
        elif net_name=='unet':
           # input_shape = [512,512,3]
            input_shape=input_shape_t
        elif net_name=='unet_height':
           # input_shape = [512,512,3]
            input_shape=input_shape_t
        else:
            print("the input net is not exist")
            return


        model=self.get_model(net_name,input_shape,number_class,weight_path)

        patch_weights=GetPatchWeight([input_shape[0],input_shape[1]],pad=32,last_value=0.2)
        batch_size=12
        overlap=0.50
            
        print('Number of files = ', len(img_test))
        path_size=([input_shape[0],input_shape[1]])
        for i in range(len(img_test)):
            print('Processing files:', i/len(img_test))
            ortho_path=img_test[i]
            predicted_patches=[]
            imageName = os.path.split(ortho_path)[-1]
            outName = imageName.replace('RGB', 'AGL')

            img_augs,img_size_o=GetPredictData(ortho_path,path_size,overlap_ratio=overlap)
            if 0:
                
                
                mm=0
                for imm in img_augs:
                    imm=imm.transpose(2,0,1)
                    outName_s=outName[:-4]+str(mm)+'-.tif'
                    mm=mm+1
                    tifffile.imsave(os.path.join(results_folder,outName_s),imm)
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
             #   imgBatch  = imagenet_utils.preprocess_input(img_batch)
             #   imgBatch = imgBatch / 255.0
                #patch=img_augs[get_batch_inds]
                img_batch=img_batch/125.0-1
            #                img_batch[:,:,:,-1]=img_batch[:,:,:,-1]/50.0-1
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
                   # pred = np.argmax(pred, axis=-1).astype('uint8')
                    if 0:
                        for pr in pred:
                           # pr=pred
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
            pred=patch2img_height(preds,img_size_o,patch_weights,num_class=number_class, overlap=overlap)
            tifffile.imsave(os.path.join(results_folder, outName), pred, compress=6)



def train_net(data_folder):
    detector=Track1_net(params,0)
    net='unet_height'
    pre_trained_weight_path=os.path.join('../weights/','181206-unet-height-weights.60.hdf5')
    #pre_trained_weight_path=os.path.join('./checkpoint_track3-unet_all_fixed_classweight_2','weights.80.hdf5')
    check_folder='./checkpoint_track1_height_unet_GAN/'
    single_class=False
    use_GAN=True
    #detector.train_track3_single(net_name=net,check_folder=check_folder,pre_trained_weight_path=pre_trained_weight_path)
    detector.train_track3(net_name=net,data_folder=data_folder,check_folder=check_folder,
                             pre_trained_weight_path=pre_trained_weight_path,single_class=single_class,use_GAN=use_GAN)
def test_net(data_folder,img_folder_name='imgs'):
    from track1_data_height import Merge_all_results_with_baseline,Merge_temparal_results
    detector=Track1_net(params,0)
    weight_path=os.path.join('./checkpoint_track1_height_unet_all_dynamic_patch512/', 'weights.190.hdf5')
    results_folder='../data/validate/Track1_unet_height_512patch_epoch190/'
    #weight_path=os.path.join('../weights', '181219-unet-semantic-weights.40.hdf5')
    #results_folder='C:/TrainData/Track1/train/prede_dsm'
    single_class=False
    net_name='unet_height'

 #   class_id=4
    detector.test_track3(data_folder,net_name,weight_path,results_folder,single_class=single_class,img_folder_name=img_folder_name)

    baseline_folder='G:/programs/dfc2019-master/track1/data/validate/Track1-Submission'
    all_result_folders={}
    all_result_folders[0]=results_folder
    all_result_folders[1]=baseline_folder
#    all_result_folders[2]=out_folder
#    merged_folder='G:/programs/dfc2019-master/track1/data/validate/Track1_submission_Merge_first'
    
 #   Merge_all_results_with_baseline(all_result_folders,merged_folder)
if __name__ == '__main__':

#    train_net('C:/TrainData/Track1/train/patch_512')
    test_net('C:/TrainData/Track1/test/')
#    img_folder_name='Track1-RGB'
 #   test_net('C:/TrainData/Track1/train/',img_folder_name=img_folder_name)

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