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
Dataset_path='G:/DataSet/ISPRS_semantic_labeling_Vaihingen/output/'
RGB_folder='img_out/'
GTS_folder='gts_out/'
H_folder='dsm_out/'
RGBH_folder = 'rgbh_out/'

def normalization(X):

    return X / 127.5 - 1


def inverse_normalization(X):

    return (X + 1.) / 2.


def get_nb_patch(img_dim, patch_size, image_data_format):

    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"

    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
#        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])

    elif image_data_format == "channels_last":
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
  #      img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def extract_patches(X, image_data_format, patch_size):

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X

def extract_patches_predict(X, image_data_format, patch_size):

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)
    #list_X=list_X[0,:,:,:]
    return list_X

def load_all_image(dset, image_data_format):
    import os
    from os import walk
    import string
    imgs = []
    gts=[]
    dsm=[]
    for (dirpath, dirnames, filenames) in walk(dset):
        for img in filenames:
            img_format=img[-3:]
            if img_format==image_data_format:
               gts.append(img)
               imgs.append('sub_img'+img[7:])
               dsm.append('sub_dsm'+img[7:])
    return imgs, dsm,gts, imgs[1:1000],dsm[1:1000], gts[1:1000]
#    data=tf.gfile.Glob(os.path.join(params.Dataset_path,params.GTS_folder,'*.tif')) #列出目录下的所有文件和目录

    

def load_data(dset, image_data_format):

    with h5py.File("../../data/processed/%s_data.h5" % dset, "r") as hf:

        X_full_train = hf["train_data_full"][:].astype(np.float32)
        X_full_train = normalization(X_full_train)

        X_sketch_train = hf["train_data_sketch"][:].astype(np.float32)
        X_sketch_train = normalization(X_sketch_train)

        if image_data_format == "channels_last":
            X_full_train = X_full_train.transpose(0, 2, 3, 1)
            X_sketch_train = X_sketch_train.transpose(0, 2, 3, 1)

        X_full_val = hf["val_data_full"][:].astype(np.float32)
        X_full_val = normalization(X_full_val)

        X_sketch_val = hf["val_data_sketch"][:].astype(np.float32)
        X_sketch_val = normalization(X_sketch_val)

        if image_data_format == "channels_last":
            X_full_val = X_full_val.transpose(0, 2, 3, 1)
            X_sketch_val = X_sketch_val.transpose(0, 2, 3, 1)

        return X_full_train, X_sketch_train, X_full_val, X_sketch_val

def load_image_data(img_names, dsm_names,label_names):
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
#    import tifffile as tiff
    imgs=[]
    labels=[]
    for index  in range(len(img_names)):
        img_path=Dataset_path+RGB_folder+img_names[index]
        lalel_path=Dataset_path+GTS_folder+label_names[index]
        dsm_path=Dataset_path+H_folder+dsm_names[index]
        rgb = cv2.imread(img_path)
        dsm = cv2.imread(dsm_path,-1)
        rgbh = cv2.merge((rgb,dsm))
        gts = cv2.imread(lalel_path,1)

        #cv2.imshow('rgb TIFF', rgb)
        #cv2.imshow('gts TIFF', gts)
        #cv2.imshow('dsm TIFF', dsm)
        #cv2.waitKey()

        imgs.append(rgbh)
        labels.append(gts)
    return imgs,labels
def gen_batch(X1, X2,X3, batch_size):

    while True:
        idx = np.random.choice(len(X1), batch_size, replace=False)
        x=[]
        dsm=[]
        y=[]
        for id in idx:
            x.append(X1[id])
            y.append(X3[id])
            dsm.append(X2[id])
        imgs,labels =load_image_data(x, dsm,y)
        y_train = np.array(labels, np.float32)/255.
        x_train = np.array(imgs, np.float32)/255.

        yield x_train, y_train


def get_disc_batch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, nb_patch,label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_sketch_batch)
        #import keras.backend as K
        #import tensorflow as tf
        ##X_disc=softmax(X_disc, axis=-1)
        ##X_disc = K.reshape(X_disc, (-1, K.int_shape(tf.convert_to_tensor(X_disc))[-1]))
        #X_disc = tf.nn.log_softmax(X_disc)
        y_disc = np.zeros(( X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_full_batch[:,:,:,0]
        X_disc=X_disc[:, :,:,np.newaxis]
        y_disc = np.zeros((X_disc.shape[0],2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:,1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    #x=Concatenate()([side1,side2,side3,side4,side5])
    #X_disc=Concatenate(axis=-1)([X_sketch_batch,X_sketch_batch]) #################
    X_sketch_batch=X_sketch_batch/125.0-1
    X_disc=X_disc/25.0-1
    X_disc=np.concatenate((X_disc,X_sketch_batch), axis=-1)
    X_disc = extract_patches(X_disc, image_data_format, patch_size)
    #X_disc=np.array(X_disc,np.float32);
    #y_disc = Concatenate(axis=-1)(y_disc)
    return X_disc, y_disc

def get_disc_batch_patch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    #if batch_counter % 2 == 0:
        # Produce an output
    x_prediction = generator_model.predict(X_sketch_batch)
    x_label=X_full_batch
    x_input=X_sketch_batch

    x_dis,y_dis=GenerateDisTrainBatch(x_prediction,x_label,x_input,patch_size,image_data_format)
        

    return x_dis, y_dis
def get_disc_batch_patch_refine(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    #if batch_counter % 2 == 0:
        # Produce an output
    x_prediction = generator_model.predict(X_sketch_batch)
    x_label=X_full_batch
    x_input=X_sketch_batch

    x_refine, y_refine, x_dis,y_dis=GenerateDisTrainBatch_refine(x_prediction,x_label,x_input,patch_size,image_data_format)
        

    return x_refine, y_refine, x_dis, y_dis

def GenerateDisTrainBatch_refine(x_prediction,x_label,x_input,patch_size,image_data_format):
    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        x_prediction = x_prediction.transpose(0,2,3,1)
        x_input = x_input.transpose(0,2,3,1)

    list_X = []
    list_Y=[]
    list_x_r=[]
    list_y_r=[]
    
    input_label=x_label
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(x_input.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(x_input.shape[2] // patch_size[1])]
    import random
    aa=random.randint(0,10)/1000
    x_label[:,:,:,0]=np.where(x_label[:,:,:,0]==1,1-aa,aa)
    x_label[:,:,:,1]=np.where(x_label[:,:,:,1]==1,1-aa,aa)
    X_p=np.concatenate((x_prediction,x_input), axis=-1)
    X_t=np.concatenate((x_label,x_input), axis=-1)
    for i in range(x_input.shape[0]):
        x_patch_r=0
        Y_patch_r=0
        largest_loss=0
        for row_idx in list_row_idx:
            for col_idx in list_col_idx:        
                p_patch=X_p[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                t_patch=X_t[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                loss_=ComparPatch(p_patch,t_patch)
               
                list_X.append(p_patch)
                list_X.append(t_patch)               
                list_Y.append(loss_)
                list_Y.append(0)
                if (loss_>largest_loss):
                    largest_loss=loss_
                    x_patch_r=x_input[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                    Y_patch_r=input_label[i, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                    x_patch_r=cv2.resize(x_patch_r,(320,320))
                    Y_patch_r=cv2.resize(Y_patch_r,(320,320))
        list_x_r.append(x_patch_r)
        list_y_r.append(Y_patch_r)

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

    idx = np.random.permutation(len(list_x_r))
    list_x_r_=[]
    list_y_r_=[]
    for id in idx:
        list_x_r_.append(list_x_r[id])
        list_y_r_.append(list_y_r[id])

    x_r = np.array(list_x_r_, np.float32) 
    y_r = np.array(list_y_r_, np.float32)
    return x_r, y_r, x_dis, y_dis

def GenerateDisTrainBatch(x_prediction,x_label,x_input,patch_size,image_data_format):
    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        x_prediction = x_prediction.transpose(0,2,3,1)
        x_input = x_input.transpose(0,2,3,1)

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
    
    onehot_labels=t_patch[:,:,0:2]
    y_ = np.reshape(onehot_labels, (-1, onehot_labels.shape[-1]))
    #onehot_labels=np.squeeze(onehot_labels.flatten())

    logits=p_patch[:,:,0:2]
    x_ = np.reshape(logits, (-1, logits.shape[-1]))
    #logits=np.squeeze(logits.flatten())
    #loss=cross_entropy(x_,y_)

    loss=weighted_cross_entropy_loss(x_,y_)
    return loss

def weighted_cross_entropy_loss(logits,onehot_labels):
    """scale loss based on class weights
    """
    # compute weights based on their frequencies
    beta=np.sum(onehot_labels[:,1])
    beta=beta/onehot_labels.shape[0]
    class_weights=[beta,1-beta];
    m = onehot_labels.shape[0]
    #p = softmax(X)
    y=onehot_labels.argmax(axis=-1)
    weights=[class_weights[i] for i in y]
    log_likelihood = -np.log(logits[range(m),y])
    log_likelihood=np.dot(log_likelihood,weights)
    loss = np.sum(log_likelihood)/(m/2.0)
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
def plot_generated_batch(X_full, X_sketch, generator_model, batch_size, image_data_format, suffix,epoch):
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt

    # Generate images
    X_gen = generator_model.predict(X_sketch)

    #X_sketch = inverse_normalization(X_sketch)
    #X_full = inverse_normalization(X_full)
    #X_gen = inverse_normalization(X_gen)
    
    Xs = np.uint8(X_sketch[0,:,:,:3]*255)
    Xg = np.uint8(X_gen[0]*255)
    Xr = np.uint8(X_full[0]*255)

    #Xs = Xs[0]
    #Xg = Xg[0]
    #Xr = Xr[0]   
    cv2.imwrite('gts_e%s.png'% epoch,Xr)
    cv2.imwrite('output_e%s.png'% epoch,Xg)
    cv2.imwrite('input_e%s.png'% epoch,Xs)

    #cv2.imshow('xs',Xs)
    #cv2.imshow('Xg',Xg)
    #cv2.imshow('Xr',Xr)
 
    #cv2.waitKey()


    #import tensorflow as tf
    #Xs=tf.image.convert_image_dtype(Xs, dtype=tf.uint8, saturate=True)
    #Xg=tf.image.convert_image_dtype(Xg, dtype=tf.uint8, saturate=True)
    #Xr=tf.image.convert_image_dtype(Xr, dtype=tf.uint8, saturate=True)
#    from PIL import Image
#    img = Image.fromarray(Xs, 'RGB')
#    img=img.transpose(2,1,0)
#    img.save('my_input_e%s.png'% epoch)
#    img.show()

#    img = Image.fromarray(Xg, 'RGB')
#    img=img.transpose(2,1,0)
#    img.save('my_output_e%s.png'% epoch)
#    img.show()

#    img = Image.fromarray(Xr, 'RGB')
#    img=img.transpose(2,1,0)
#    img.save('my_gts_e%s.png'% epoch)
#    img.show()

#    if image_data_format == "channels_last":
#        X = np.concatenate((Xs, Xg, Xr), axis=0)
#      #  X = np.concatenate((Xg, Xr), axis=0)
#        list_rows = []
#        for i in range(int(X.shape[0] // 4)):
#            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
#            list_rows.append(Xr)

#        Xr = np.concatenate(list_rows, axis=0)

#    if image_data_format == "channels_first":
#        X = np.concatenate((Xs, Xg, Xr), axis=0)
#        list_rows = []
#        for i in range(int(X.shape[0] // 4)):
#            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
#            list_rows.append(Xr)

#        Xr = np.concatenate(list_rows, axis=1)
#        Xr = Xr.transpose(1,2,0)
##    Xr=Xr255
#    if Xr.shape[-1] == 1:
#        plt.imshow(Xr[:, :, 0], cmap="gray")
#    else:
#        plt.imshow(Xr)
#    plt.axis("off")
#    plt.savefig("../../figures/current_batch_%s.png" % suffix)
#    plt.clf()
#    plt.close()
