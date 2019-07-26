from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import h5py
from keras.layers import *
import matplotlib.pylab as plt
import cv2
from keras.engine import Layer
import tensorflow as tf
import keras.backend as K
import math
import PIL.Image     as Image
try:
    from itertools import izip
except ImportError:
    izip = zip

import multiprocessing as mp
   

def normalization(X):

    return X / 127.5 - 1


def inverse_normalization(X):

    return (X + 1.) / 2.

import numpy as np

def softmax(x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)

def get_disc_batch_patch_refine(X_full_batch, X_sketch_batch, generator_model,  patch_size,
                   train_gen, dis_model='class',dis_input_type='softmax',image_data_format='channels_last',label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    #if batch_counter % 2 == 0:
        # Produce an output
    x_prediction = generator_model.predict(X_sketch_batch)
    x_prediction=softmax(x_prediction)
    x_label=X_full_batch
    x_input=X_sketch_batch

    if train_gen:
        x_refine, y_refine, x_dis,y_dis=GenerateDisTrainBatch_refine(x_prediction,x_label,x_input,patch_size,train_gen,dis_model,dis_input_type)
        return x_refine, y_refine, x_dis, y_dis
    else:
         x_dis,y_dis=GenerateDisTrainBatch_refine(x_prediction,x_label,x_input,patch_size,train_gen,dis_model,dis_input_type)
         return x_dis, y_dis

def get_disc_batch_patch_refine_mp(X_full_batch, X_sketch_batch, generator_model,  patch_size,
                   train_gen, dis_model='class',dis_input_type='softmax',image_data_format='channels_last',label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    #if batch_counter % 2 == 0:
        # Produce an output
    x_prediction = generator_model.predict(X_sketch_batch)
    x_prediction=softmax(x_prediction)
    x_label=X_full_batch
    x_input=X_sketch_batch
    pool=mp.Pool()
    
    x_r=[]
    y_r=[]
    x_d=[]
    y_d=[]
    x_pp=[]
    x_lab=[]
    x_inp=[]
    #aa=x_prediction.tolist()
    for i in range(x_prediction.shape[0]):
        aa=x_prediction[i,:,:,:]
        aa=aa[np.newaxis,:]
        x_pp.append(aa)

        bb=x_label[i,:,:,:]
        bb=bb[np.newaxis,:]
        x_lab.append(bb)

        bb=x_input[i,:,:,:]
        bb=bb[np.newaxis,:]        
        x_inp.append(bb)


    img_label=zip(x_pp,x_lab,x_inp)
    #res=GenerateDisTrainBatch_refine(x_pp[0],x_lab[0],x_inp[0])
    res=pool.starmap(GenerateDisTrainBatch_refine,img_label)
    if train_gen:
        
        for x_refine, y_refine, x_dis,y_dis in res:
            #x_r=np.concatenate((x_r,x_refine), axis=0)
            #y_r=np.concatenate((y_r,y_refine), axis=0)
            #x_d=np.concatenate((x_d,x_dis), axis=0)
            #y_d=np.concatenate((y_d,y_dis), axis=0)
                x_r.append(np.squeeze(x_refine))
                y_r.append(np.squeeze(y_refine))
                x_d.append(np.squeeze(x_dis))
                y_d.append(np.squeeze(y_dis))
        x_r = np.array(x_r, np.float32)
        y_r = np.array(y_r, np.float32)
        y_r=y_r[:,:,:,np.newaxis] 
        dd=y_d[-1]
        cc=x_d[-1]
        for i in range(len(y_d)-1):
    
           x_d_ = np.concatenate((cc,x_d[i]),axis=0)
           y_d_ = np.concatenate((dd, y_d[i]),axis=0)
        return x_r, y_r,x_d_,y_d_

    else:
         for x_dis,y_dis in res:
                x_d.append(x_dis)
                y_d.append(y_dis)
         x_d = np.array(x_d, np.float32)
         y_d = np.array(y_d, np.float32)
         return x_d, y_d


def GenerateDisTrainBatch_refine(x_prediction,x_label,x_input,patch_size=[160,160],train_gen=True, dis_model='regress',dis_input_type='label',image_data_format='channels_last'):
    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        x_prediction = x_prediction.transpose(0,2,3,1)
        x_input = x_input.transpose(0,2,3,1)

    list_X = []
    list_Y=[]
    list_x_r=[]
    list_y_r=[]
    
    input_label=x_label
        #list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(x_input.shape[1] // patch_size[0])]

    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(x_input.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(x_input.shape[2] // patch_size[1])]
    import random
    #aa=random.randint(0,10)/1000
    x_truth=np.zeros((x_prediction.shape[0],x_prediction.shape[1],x_prediction.shape[2],x_prediction.shape[3]))


    label_onehot = np.zeros(x_truth.shape, dtype=np.float32)
    for i in range(label_onehot.shape[0]):
        for j in range(label_onehot.shape[1]):
            for k in range(label_onehot.shape[2]):
                id=np.int(x_label[i,j,k,0])
                if id>=x_prediction.shape[-1]:
                    continue;
                label_onehot[i,j,k,id]=1

    #for i in range(label_onehot.shape[-1]):
    #    aa=random.randint(0,10)/10000
    #    x_truth[:,:,:,i]=np.where(label_onehot[:,:,:,i]==1,1-aa,aa)  ######### need to be improved
    x_truth=label_onehot
    
    
    if dis_input_type=='softmax':
        X_p=np.concatenate((x_prediction,x_input), axis=-1)
        X_t=np.concatenate((x_truth,x_input), axis=-1)
    elif dis_input_type=='label':
        label_id=np.arange(x_prediction.shape[3])
        predict_label=np.argmax(x_prediction,axis=-1).astype(np.uint8)
        #predict_label_=predict_label/x_prediction.shape[3]-1
        predict_label= predict_label[:, :,:,np.newaxis]
        X_p=np.concatenate((predict_label,x_input), axis=-1)
        X_t=np.concatenate((x_label,x_input), axis=-1)

    for i in range(x_input.shape[0]):
        x_patch_r=0
        Y_patch_r=0
        largest_loss=0
        for row_idx in list_row_idx:
            for col_idx in list_col_idx:        
                p_patch=X_p[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                t_patch=X_t[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                list_X.append(p_patch)
                list_X.append(t_patch)  
                if train_gen or dis_model=='regress':
                    if dis_input_type=='label':
                        pre_patch=predict_label[i,row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]]
                        tru_patch=x_label[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1],0]                       
                        loss_=1-evaluatePair(pre_patch,tru_patch,label_id=label_id)
                    else:
                        loss_=ComparPatch(x_prediction[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :],label_onehot[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
                if dis_model=='regress':
                    list_Y.append(loss_)
                    list_Y.append(0)
                else:
                    list_Y.append([1,0])
                    list_Y.append([0,1])
                if train_gen:
                    if (loss_>largest_loss):
                        largest_loss=loss_
                        x_patch_r=x_input[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                        Y_patch_r=x_label[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
        if train_gen:
            x_patch_r=cv2.resize(x_patch_r,(x_input.shape[1],x_input.shape[2]))
            Y_patch_r=cv2.resize(Y_patch_r,(x_input.shape[1],x_input.shape[2]),interpolation=cv2.INTER_NEAREST)
            list_x_r.append(x_patch_r)
            list_y_r.append(Y_patch_r)


    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)
    #list_X=list_X[0,:,:,:]
    import random
    idx = np.random.permutation(len(list_X))
    list_X_=[]
    list_Y_=[]
    for id in idx:
        list_X_.append(list_X[id])
        list_Y_.append(list_Y[id])
    #list_X_=random.shuffle(list_X)  
    #list_Y_=random.shuffle(list_X)  
    x_dis = np.array(list_X_, np.float32) 
    y_dis = np.array(list_Y_, np.float32)
    if train_gen:
        idx = np.random.permutation(len(list_x_r))
        list_x_r_=[]
        list_y_r_=[]
        for id in idx:
            list_x_r_.append(list_x_r[id])
            list_y_r_.append(list_y_r[id])

        x_r = np.array(list_x_r_, np.float32) 
        y_r = np.array(list_y_r_, np.float32)
        y_r = y_r[:, :,:,np.newaxis]
        return x_r, y_r, x_dis, y_dis
    else:
        return x_dis, y_dis
def get_disc_batch_patch(X_full_batch, X_sketch_batch, generator_model,  patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    #if batch_counter % 2 == 0:
        # Produce an output
    x_prediction = generator_model.predict(X_sketch_batch)
    ss=np.argwhere(np.isnan(x_prediction))
    if ss.shape[0]>0:
        #ssss=aa[ss]
        aa=0
    ss=np.argwhere(np.isinf(x_prediction))
    if ss.shape[0]>0:
        #ssss=aa[ss]
        aa=0

    for  i in range(x_prediction.shape[-1]):
        name='aaa'+str(i)+'.txt'
        np.savetxt(name,x_prediction[0,:,:,i])
    x_prediction=softmax(x_prediction)
    for  i in range(x_prediction.shape[-1]):
        name='aaabb'+str(i)+'.txt'
        np.savetxt(name,x_prediction[0,:,:,i])
    x_label=X_full_batch
    x_input=X_sketch_batch

    x_dis,y_dis=GenerateDisTrainBatch(x_prediction,x_label,x_input,patch_size,image_data_format)
        

    return  x_dis, y_dis

def GenerateDisTrainBatch(x_prediction,x_label,x_input,patch_size,image_data_format):
    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        x_prediction = x_prediction.transpose(0,2,3,1)
        x_input = x_input.transpose(0,2,3,1)

    list_X = []
    list_Y=[]

    #x_prediction=softmax_(x_prediction)
    input_label=x_label
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(x_input.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(x_input.shape[2] // patch_size[1])]
    import random
    #aa=random.randint(0,10)/1000
    x_truth=np.zeros((x_prediction.shape[0],x_prediction.shape[1],x_prediction.shape[2],x_prediction.shape[3]))


    label_onehot = np.zeros(x_truth.shape, dtype=np.float32)
    for i in range(label_onehot.shape[0]):
        for j in range(label_onehot.shape[1]):
            for k in range(label_onehot.shape[2]):
                id=np.int(x_label[i,j,k,0])
                if id>=x_prediction.shape[-1]:
                    continue;
                label_onehot[i,j,k,id]=1

    for i in range(label_onehot.shape[-1]):
        aa=random.randint(0,10)/10000
        x_truth[:,:,:,i]=np.where(label_onehot[:,:,:,i]==1,1-aa,aa)  ######### need to be improved
    
    X_p=np.concatenate((x_prediction,x_input), axis=-1)
    X_t=np.concatenate((x_truth,x_input), axis=-1)
    for i in range(x_input.shape[0]):
        for row_idx in list_row_idx:
            for col_idx in list_col_idx:        
                p_patch=X_p[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                t_patch=X_t[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                #np.savetxt('aaa.txt',p_patch)
                #np.savetxt('bb.txt',t_patch)
                loss_=ComparPatch(x_prediction[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :],label_onehot[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
                
                list_X.append(p_patch)
                list_X.append(t_patch)               
                list_Y.append(loss_)
                list_Y.append(0)

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)
    #list_X=list_X[0,:,:,:]
    import random
    idx = np.random.permutation(len(list_X))
    list_X_=[]
    list_Y_=[]
    for id in idx:
        list_X_.append(list_X[id])
        list_Y_.append(list_Y[id])
    #list_X_=random.shuffle(list_X)  
    #list_Y_=random.shuffle(list_X)  
    x_dis = np.array(list_X_, np.float32) 
    y_dis = np.array(list_Y_, np.float32)

    return x_dis, y_dis
def GenerateDisTrainBatch_(x_prediction,x_label,x_input,patch_size,image_data_format):
    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        x_prediction = x_prediction.transpose(0,2,3,1)
        x_input = x_input.transpose(0,2,3,1)
    x_prediction=softmax_(x_prediction)
    list_X = []
    list_Y=[]
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(x_input.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(x_input.shape[2] // patch_size[1])]
    import random
    aa=random.randint(0,10)/1000


    x_label[:,:,:,0]=np.where(x_label[:,:,:,0]==1,1-aa,aa)
    x_label[:,:,:,1]=np.where(x_label[:,:,:,1]==1,1-aa,aa)
    X_p=np.concatenate((x_prediction,x_input), axis=-1)
    X_t=np.concatenate((x_label,x_input), axis=-1)
    for i in range(x_input.shape[0]):
        for row_idx in list_row_idx:
            for col_idx in list_col_idx:        
                p_patch=X_p[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                t_patch=X_t[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                loss_=ComparPatch(p_patch,t_patch)

                list_X.append(p_patch)
                list_X.append(t_patch)               
                list_Y.append(loss_)
                list_Y.append(0)

                #list_Y.append(0)
    #for i in range(x_input.shape[0]):
    #    for row_idx in list_row_idx:
    #         for col_idx in list_col_idx:
           
    #            #aa=random.randint(0,10)/1000.0
    #            list_X.append(X_t[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    #            list_Y.append([0,1])
    #            #list_Y.append(1)

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)
    #list_X=list_X[0,:,:,:]
    import random
    idx = np.random.permutation(len(list_X))
    list_X_=[]
    list_Y_=[]
    for id in idx:
        list_X_.append(list_X[id])
        list_Y_.append(list_Y[id])
    #list_X_=random.shuffle(list_X)  
    #list_Y_=random.shuffle(list_X)  
    x_dis = np.array(list_X_, np.float32) 
    y_dis = np.array(list_Y_, np.float32)
    return x_dis, y_dis

def ComparPatch(p_patch,t_patch):
    
    #onehot_labels=t_patch[:,:,0:2]
    y_ = np.reshape(t_patch, (-1, t_patch.shape[-1]))
    #onehot_labels=np.squeeze(onehot_labels.flatten())

    #logits=p_patch[:,:,0:2]
    x_ = np.reshape(p_patch, (-1, p_patch.shape[-1]))
    #logits=np.squeeze(logits.flatten())
    #loss=cross_entropy(x_,y_)

    loss=weighted_cross_entropy_loss(x_,y_)
    return loss
def sparse_crossentropy_ignoring_last_label(y_pred,y_true ):

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def weighted_cross_entropy_loss(logits,onehot_labels):
    """scale loss based on class weights
    """
    # compute weights based on their frequencies
    #beta=np.sum(onehot_labels[:,1])
    #beta=beta/onehot_labels.shape[0]
    #class_weights=[beta,1-beta];
    m = onehot_labels.shape[0]
    #p = softmax(X)
    y=onehot_labels.argmax(axis=-1)
    #y=onehot_labels
    #weights=[class_weights[i] for i in y]
    #for value in logits[range(m),y]:
    #    if (value<0.000000001 ):
    #        cc=0
    values=logits[range(m),y]
    values[np.where(np.isnan(values))] = 0.000000001
    #for value in values:
    #    if (value<0.00000000001 ):
    #        cc=0
    log_likelihood = -np.log(values)
    
    #log_likelihood=np.dot(log_likelihood,weights)
    loss = np.sum(log_likelihood)/(m/2.0)
    ss=np.argwhere(np.isnan(logits))
    if ss.shape[0]>0:
        ssss=values[ss]
        aa=0
    ss=np.argwhere(np.isinf(logits))
    if ss.shape[0]>0:
        ssss=values[ss]
        aa=0
    if np.isnan(loss):
        loss=0.0001
    if np.isinf(loss):
        loss=0.0001
    return loss


def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    #p = softmax(X)
    y=y.argmax(axis=-1)
    # We use multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(X[range(m),y])
    loss = np.sum(log_likelihood) / (m)
    return loss


def evaluatePair(predictionNp, groundTruthNp,label_id=[7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33,0]):
    #label_id=[7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33,0]
    imgWidth  = predictionNp.shape[0]
    imgHeight = predictionNp.shape[1]
    nbPixels  = imgWidth*imgHeight
    n_correct=0
    # Evaluate images
    #if (0):
    #    # using cython
    #    confMatrix = addToConfusionMatrix.cEvaluatePair(predictionNp, groundTruthNp, confMatrix, args.evalLabels)
    #else:
        # the slower python way
    for (groundTruthImgPixels,predictionImgPixels) in izip(predictionNp,groundTruthNp):
        for (groundTruthImgPixel,predictionImgPixel) in izip(groundTruthImgPixels,predictionImgPixels):
            if (not groundTruthImgPixel in label_id):
                groundTruthImgPixel=0
            if ( groundTruthImgPixel==predictionImgPixel):
                    n_correct+=1


    return n_correct/nbPixels


def GenerateDisTrainBatch_p(x_prediction,x_input,patch_size,image_data_format):
    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        x_prediction = x_prediction.transpose(0,2,3,1)
        x_input = x_input.transpose(0,2,3,1)

    list_X = []

    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(x_input.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(x_input.shape[2] // patch_size[1])]

    X_p=np.concatenate((x_prediction,x_input), axis=-1)
    for i in range(x_input.shape[0]):
        for row_idx in list_row_idx:
            for col_idx in list_col_idx:            
                list_X.append(X_p[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

                #list_Y.append(0)

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)
    #list_X=list_X[0,:,:,:]

    x_dis = np.array(list_X, np.float32) 

    return x_dis

def Calculate_patch_loss(prediction_list, groundtruth_list, patch_size, output_folder, loss_fun='crossentropy'):
    print("Evaluating {} pairs of images...".format(len(prediction_list)))
    for i in range(len(prediction_list)):
        scores=[]
        predictionImgFileName = prediction_list[i]
        groundTruthImgFileName = groundtruth_list[i]

        
        try:
            predictionImg = Image.open(predictionImgFileName)
            predictionNp  = np.array(predictionImg)
        except:
            printError("Unable to load " + predictionImgFileName)
        try:
            groundTruthImg = Image.open(groundTruthImgFileName)
            groundTruthNp = np.array(groundTruthImg)
        except:
            printError("Unable to load " + groundTruthImgFileName)
 
        # Check for equal image sizes
        if (predictionImg.size[0] != groundTruthImg.size[0]):
            printError("Image widths of " + predictionImgFileName + " and " + groundTruthImgFileName + " are not equal.")
        if (predictionImg.size[1] != groundTruthImg.size[1]):
            printError("Image heights of " + predictionImgFileName + " and " + groundTruthImgFileName + " are not equal.")
        if ( len(predictionNp.shape) != 2 ):
            printError("Predicted image has multiple channels.")
################################
        ixx=predictionImgFileName.rfind('\\')
        img_name=predictionImgFileName[ixx+1:-4]

        list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(predictionImg.size[1] // patch_size[0])]
        list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(predictionImg.size[0] // patch_size[1])]
        for row_idx in list_row_idx:
            for col_idx in list_col_idx:   
                score=evaluatePair(predictionNp[row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]],groundTruthNp[row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])
                scores.append(score)
        #scores=(scores-min(scores))
        #scores=scores/max(scores)*255
            #print(scores2)
        scores=np.array(scores)
        scores=(1-scores)*255
            #print(scores)
        score_map=np.reshape(scores,(len(list_row_idx),len(list_col_idx),1))
        score_map=cv2.resize(score_map,(groundTruthImg.size[0],groundTruthImg.size[1]),interpolation=cv2.INTER_NEAREST)
        score_map = np.expand_dims(score_map, axis=-1)
        score_map= score_map.astype(int)
        dis_img_path=os.path.join(output_folder, img_name + '_patch-score-t.png')
        cv2.imwrite(dis_img_path,score_map)
        dis_txt_path=os.path.join(output_folder, img_name + '_patch-score_t.txt')
        fh = open(dis_txt_path, 'w')
        fh.write(str(scores/255.0))
        fh.close()
        print("\rImages Processed: {}".format(i+1), end=' ')




if __name__ == "__main__":
    img_files=[]
    label_files=[]

    import os
    file='G:/programs/FCN_GAN/src/prediction_retrain_100_/_prediction-list.txt'
    fp = open(file)
    lines = fp.readlines()
    fp.close()

    prediction_folder='G:/programs/FCN_GAN/src/prediction_retrain_100_'
    nb_sample = len(lines)
    for line in lines:
            line = line.strip('\n')    
            lable_p_path=os.path.join(prediction_folder, line)
            img_files.append(lable_p_path)
    
    file='G:/DataSet/CityScapes_cl/resized/eval_label_list.txt'
    fp = open(file)
    lines = fp.readlines()
    fp.close()
    for line in lines:
            line = line.strip('\n')           
            label_files.append(line)

    num_class=20
    train_imgs=img_files[0:800]
    train_labels=label_files[0:800]
    val_imgs=img_files[800:1000]
    val_labels=label_files[800:1000]

    # evaluate
    patch_size=[128,256]
    output_folder=prediction_folder
    Calculate_patch_loss(img_files, val_labels, patch_size,output_folder)

    