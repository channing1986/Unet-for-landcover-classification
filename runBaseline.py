__author__ = 'jhuapl'
__version__ = 0.1

import sys
import os
import numpy as np
import params
from dfcBaseline import DFCBaseline
from dataFunctions import parse_args
from dataFunctions import *

def main(argv):
    # training, mode = parse_args(argv, params)
    training=0
    mode=0
    os.environ["CUDA_VISIBLE_DEVICES"]=params.GPUS
    model = DFCBaseline(params=params, mode=mode)
    if training:
        try:
            model.train()
        except ValueError as error:
            print(error, file=sys.stderr)
    else:
       # model.test()
        model.test_patch()

def train_track3(argv):
    os.environ["CUDA_VISIBLE_DEVICES"]=params.GPUS
    from track3_data import load_all_data_files,input_generator_RGBH
    img_train, dsm_train, lable_train, img_val, dsm_val, label_val=load_all_data_files()
    
    num_training_sample=len(img_train)
    batch_size=20
    n_batch_per_epoch=num_training_sample//batch_size

    num_val_sample=len(img_val)
    n_batch_per_epoch_val=num_val_sample//batch_size

    input_shape = [256,256,9]
    nb_epoch=200

    train_generator=input_generator_RGBH(img_train,dsm_train, lable_train, batch_size)
    val_generator = input_generator_RGBH(img_val,dsm_val,label_val,batch_size)

    from segmentation_models import UnetRegressor,Unet,pspnet
    from keras.layers import Input
    from keras.models import load_model
    from keras import backend as K
    from keras.optimizers import Adam, SGD
    from keras.callbacks import ModelCheckpoint,CSVLogger
    input_tensor = Input(shape=(input_shape[0],input_shape[1],input_shape[2]))
    model = Unet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=params.BACKBONE, 
                         encoder_weights=None, classes=params.NUM_CATEGORIES)
    #loss=params.SEMANTIC_LOSS
    loss=my_weighted_loss
    optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(optimizer, loss=loss)
    model.load_weights(os.path.join('./checkpoint_track3-1/', 'weights.80.hdf5'))
    model.summary()
    csv_logger = CSVLogger(os.path.join(params.CHECKPOINT_DIR,'train.csv'))

    checkpoint = ModelCheckpoint(filepath=params.CHECKPOINT_PATH, monitor='loss', verbose=1, save_best_only=False,
                                    save_weights_only=True, mode='auto', period=params.MODEL_SAVE_PERIOD)
    callbacks=[csv_logger,  checkpoint]
    class_weight = {0: 1.,
                1: 2.,
                2: 2.,
                3: 60.,
                4: 70.}
    model.fit_generator(train_generator, steps_per_epoch=n_batch_per_epoch,
                        validation_data=val_generator, validation_steps=n_batch_per_epoch_val,
                        epochs=nb_epoch, callbacks=callbacks)

def test_track3(argv):
    os.environ["CUDA_VISIBLE_DEVICES"]=params.GPUS
    from track3_data import load_all_data_test,GetPredictData
    from DataAugment import goback_auger_p,patch2img,GetPatchWeight
    from dataFunctions import convert_labels
    ortho_txt='C:/TrainData/Track3/Test/test_orthos.txt'
    dsm_txt='C:/TrainData/Track3/Test/test_dsm.txt'
    img_test, dsm_test=load_all_data_test(ortho_txt,dsm_txt)
 #   img_test=img_test[95:120]
 #   dsm_test=dsm_test[95:120]
    print('Number of files = ', len(img_test))
###########load model
    input_shape = [256,256,9]
    from segmentation_models import Unet
    from keras.layers import Input
    from keras.optimizers import Adam
    input_tensor = Input(shape=(input_shape[0],input_shape[1],input_shape[2]))
    model = Unet(input_shape=input_shape, input_tensor=input_tensor, backbone_name=params.BACKBONE, 
                         encoder_weights=None, classes=params.NUM_CATEGORIES)
    loss=params.SEMANTIC_LOSS
    optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer, loss=loss)
    model.load_weights(os.path.join('./checkpoint_track3-4/', 'weights.80.hdf5'))
    model.summary()
#####
        
    patch_weights=GetPatchWeight([input_shape[0],input_shape[1]])
    for i in range(len(img_test)):
        ortho_path = img_test[i]
        dsm_path=dsm_test[i]
        predicted_patches=[]
        imageName = os.path.split(ortho_path)[-1]
        outName = imageName.replace('ortho', 'CLS')
        img_augs=GetPredictData(ortho_path,dsm_path)
        
        from track3_data import GRID
        gdal_reader=GRID()
        if 0:
            
            
            mm=1
            for imm in img_augs:
                imm=imm.transpose(2,0,1)
                outName_s=outName[:-4]+str(mm)+'-.tif'
                mm=mm+1
                gdal_reader.write_img(os.path.join(params.OUTPUT_DIR_TRACK3,outName_s),[],[],imm[0:3,:,:])
                #tifffile.imsave(os.path.join(params.OUTPUT_DIR_TRACK3,outName_s),imm[:,:,0:3],compress=6)
         #   img=load_img(imgPath)
        
        count=0
        for patch in img_augs:
            #img=np.array(patch,np.float32)
            patch[:,:,0:-1]=patch[:,:,0:-1]/125.0-1
            patch[:,:,-1]=patch[:,:,-1]/50.0-1
            img = np.expand_dims(patch, axis=0).astype('float32')
            pred = model.predict(img)
        
        #  if self.mode == self.params.SEMANTIC_MODE:
            if params.NUM_CATEGORIES > 1:
                pred = np.argmax(pred, axis=-1).astype('uint8')
                if 0:
                    
                    outName_s=outName[:-4]+str(count)+'_888.tif'
                    count=count+1
                    tifffile.imsave(os.path.join(params.OUTPUT_DIR_TRACK3, outName_s), pred, compress=6)
            else:
                pred = (pred > params.BINARY_CONF_TH).astype('uint8')
            if params.CONVERT_LABELS:
                pred = convert_labels(pred, params, toLasStandard=True)
            predicted_patches.append(pred)
        preds=goback_auger_p(predicted_patches)

        if 0:
            mm=0
            for imm in preds:
                outName_s=outName[:-4]+str(mm)+'l-.tif'
                mm=mm+1
                tifffile.imsave(os.path.join(params.OUTPUT_DIR_TRACK3,outName_s),imm,compress=6)
        proj2,geotrans2,dsm=gdal_reader.read_img(dsm_path)
       # dsm=cv2.imread(dsm_path,0)
        img_size_o=(dsm.shape[0],dsm.shape[1])
        overlap=0.5
        
        pred=patch2img(preds,img_size_o,patch_weights,num_class=5, overlap=overlap)
        tifffile.imsave(os.path.join(params.OUTPUT_DIR_TRACK3, outName), pred, compress=6)

if __name__ == "__main__":
 #   train_track3(sys.argv)
    test_track3(sys.argv)
# python ./unets/runBaseline.py train semantic
   # main(sys.argv)
