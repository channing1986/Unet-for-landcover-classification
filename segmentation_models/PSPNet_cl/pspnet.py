#!/usr/bin/env python
from __future__ import print_function
import os
from os.path import splitext, join, isfile, isdir, basename
import argparse
import numpy as np
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json, load_model
import tensorflow as tf
from . import layers_builder as layers
from glob import glob
#from .python_utils import utils
#from python_utils.preprocessing import preprocess_img
from keras.utils.generic_utils import CustomObjectScope
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape, weights):
        self.input_shape = input_shape
        json_path = join("segmentation_models","PSPNet_cl","weights", "keras", weights + "0..0.json")
        h5_path = join("segmentation_models","PSPNet_cl","weights", "keras", weights + ".h5")
        if 'pspnet' in weights:
            if os.path.isfile(json_path) and os.path.isfile(h5_path):
                print("Keras model & weights found, loading...")
                with CustomObjectScope({'Interp': layers.Interp}):
                    with open(json_path, 'r') as file_handle:
                        self.model = model_from_json(file_handle.read())
                self.model.load_weights(h5_path,by_name=True)
            else:
                print("No Keras model & weights found, import from npy weights.")
                self.model = layers.build_pspnet(nb_classes=nb_classes,
                                                 resnet_layers=resnet_layers,
                                                 input_shape=self.input_shape)
                if os.path.isfile(h5_path):   
                    self.model.load_weights(h5_path,by_name=True)
#                    print("Writing keras model & weights")
#                    json_string = self.model.to_json()
#                    with open(json_path, 'w') as file_handle:
#                        file_handle.write(json_string)
#                    self.model.save_weights(h5_path)
                    print("Finished load weights")

                #self.set_npy_weights(weights)

        else:
            print('Load pre-trained weights')
            self.model = load_model(weights)

    def predict(self, img, flip_evaluation=False):
        """
        Predict segementation for an image.

        Arguments:
            img: must be rowsxcolsx3
        """
        h_ori, w_ori = img.shape[:2]

        # Preprocess
        img = misc.imresize(img, self.input_shape[0:2])

        img = img - DATA_MEAN
        img = img[:, :, ::-1]  # RGB => BGR
        img = img.astype('float32')
        print("Predicting...")

        probs = self.feed_forward(img, flip_evaluation)

        if img.shape[0:1] != self.input_shape[0:2]:  # upscale prediction if necessary
            h, w = probs.shape[:2]
            probs = ndimage.zoom(probs, (1. * h_ori / h, 1. * w_ori / w, 1.),
                                 order=1, prefilter=False)

        print("Finished prediction...")

        return probs

    def feed_forward(self, data, flip_evaluation=False):
        assert data.shape == (self.input_shape[0], self.input_shape[1], self.input_shape[2])

        if flip_evaluation:
            print("Predict flipped")
            input_with_flipped = np.array(
                [data, np.flip(data, axis=1)])
            prediction_with_flipped = self.model.predict(input_with_flipped)
            prediction = (prediction_with_flipped[
                          0] + np.fliplr(prediction_with_flipped[1])) / 2.0
        else:
            prediction = self.model.predict(np.expand_dims(data, 0))[0]
        return prediction

    def set_npy_weights(self, weights_path):
        npy_weights_path = join("segmentation_models","PSPNet_cl","weights", "npy", weights_path + ".npy")
        json_path = join("segmentation_models","PSPNet_cl","weights", "keras", weights_path + ".json")
        h5_path = join("segmentation_models","PSPNet_cl","weights", "keras", weights_path + ".h5")

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path, encoding='bytes').item()
        for layer in self.model.layers:
            print(layer.name)
            if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
                mean = weights[layer.name.encode()][
                    'mean'.encode()].reshape(-1)
                variance = weights[layer.name.encode()][
                    'variance'.encode()].reshape(-1)
                scale = weights[layer.name.encode()][
                    'scale'.encode()].reshape(-1)
                offset = weights[layer.name.encode()][
                    'offset'.encode()].reshape(-1)

                self.model.get_layer(layer.name).set_weights(
                    [scale, offset, mean, variance])

            elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
                try:
                    weight = weights[layer.name.encode()]['weights'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight])
                except Exception as err:
                    biases = weights[layer.name.encode()]['biases'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight,
                                                                  biases])
        print('Finished importing weights.')

        print("Writing keras model & weights")
        json_string = self.model.to_json()
        with open(json_path, 'w') as file_handle:
            file_handle.write(json_string)
        self.model.save_weights(h5_path)
        print("Finished writing Keras model & weights")


class PSPNet50(PSPNet):
    """Build a PSPNet based on a 50-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=50,
                        input_shape=input_shape, weights=weights)


class PSPNet101(PSPNet):
    """Build a PSPNet based on a 101-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=101,
                        input_shape=input_shape, weights=weights)



def Generate_list(folder):

    searchFine=os.path.join(folder,'*','*_labelIds.png')
    filesFine = glob( searchFine )
    filesFine.sort()
    #filesCoarse = glob.glob( searchCoarse )
    #filesCoarse.sort()

    # concatenate fine and coarse
    #files = filesFine + filesCoarse
    files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    if not files:
        print( "Did not find any files. Please consult the README." )

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
   
    label_list=[]
    patch_text=os.path.join(folder,'_val-list.txt')
    f= open (patch_text,"w") 
    for name in files:
            f.write(name)
            f.write('\n')
    f.close()

if __name__ == "__main__":
    

    Generate_list ('G:/DataSet/CityScapes/cityscapes/gtFine/val')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet101_voc2012',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('-i', '--input_path', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-g', '--glob_path', type=str, default=None,
                        help='Glob path for multiple images')
    parser.add_argument('-o', '--output_path', type=str, default='example_results/ade20k.jpg',
                        help='Path to output')
    parser.add_argument('--id', default="0")
    parser.add_argument('--input_size', type=int, default=500)
    parser.add_argument('-f', '--flip', type=bool, default=True,
                        help="Whether the network should predict on both image and flipped image.")

    args = parser.parse_args()
#################
    args.model='pspnet101_cityscapes'
    args.glob_path='G:/DataSet/CityScapes/cityscapes/leftImg8bit/val/'
    args.output_path='example_results_city_val/'
##############
##wsi_mask_paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.jpg'))
##wsi_mask_paths.sort()
    # Handle input and output args
    images = glob(os.path.join(args.glob_path,'*','*.png')) if args.glob_path else [args.input_path,]
    if args.glob_path:
        fn, ext = splitext(args.output_path)
        if ext:
            parser.error("output_path should be a folder for multiple file input")
        if not isdir(args.output_path):
            os.mkdir(args.output_path)

    # Predict
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        print(args)
        if not args.weights:
            if "pspnet50" in args.model:
                pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                                  weights=args.model)
            elif "pspnet101" in args.model:
                if "cityscapes" in args.model:
                    pspnet = PSPNet101(nb_classes=19, input_shape=(713, 713),
                                       weights=args.model)
                if "voc2012" in args.model:
                    pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                                       weights=args.model)

            else:
                print("Network architecture not implemented.")
        else:
            pspnet = PSPNet50(nb_classes=2, input_shape=(
                768, 480), weights=args.weights)


        label_id=[7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33,0]
        lable_color=[[128, 64,128], [244, 35,232], [ 70, 70, 70] , [102,102,156], [190,153,153],[153,153,153], 
                        [250,170, 30], [220,220,  0],  [107,142, 35], [152,251,152], [70,130,180], 
                     [220, 20, 60], [255,  0,  0], [ 0,  0,142], [0,  0, 70],[ 0, 60,100], [0, 80,100],
                     [ 0,  0,230],[119, 11, 32],[ 0,  0,  0]]
        save_path=args.output_path
        results=[]
        for i, img_path in enumerate(images):
            print("Processing image {} / {}".format(i+1,len(images)))
            img = misc.imread(img_path, mode='RGB')
            cimg = misc.imresize(img, (args.input_size, args.input_size))

            probs = pspnet.predict(img, args.flip)

            output = np.argmax(probs, axis=2)


            ixx=img_path.rfind('\\')
            img_name=img_path[ixx+1:-4]


            result=output

            #result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)
            label_map=np.zeros((result.shape[0],result.shape[1],1), dtype=np.uint8)
            label_color_img=np.zeros((result.shape[0],result.shape[1],3), dtype=np.uint8)

        
            #for i in range (num_class):
            #    label_map[np.where(result == i)] = label_id[i]
            for i in range ( result.shape[0]):
                for j in range (result.shape[1]):
                    label_color_img[i,j,:]=lable_color[result[i,j]]
                    label_map[i,j]=label_id[result[i,j]]
            #result_img = Image.fromarray(result, mode='P')
            #result_img.palette = label.palette
            #result_img = result_img.resize(label_size, resample=Image.BILINEAR)
            #result_img = result_img.crop((pad_w//2, pad_h//2, pad_w//2+img_w, pad_h//2+img_h))
            # result_img.show(title='result')
            lable_p_path=os.path.join(save_path, img_name + '_lable.png')

            if 1:
                #lables.append()
                results.append(img_name + '_lable.png')
            if 1:
                #lable_map_path=os.path.join(save_path, img_name + '_lable.png')
                label_color_img_path=os.path.join(save_path, img_name + '_color.png')
                #label_map=cv2.resize(label_map,(512,256),interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(lable_p_path,label_map)
                #label_color_img=label_color_img.transpose(2,1,0)
                #label_color_img=cv2.resize(label_color_img,(512,256),interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(label_color_img_path,cv2.cvtColor(label_color_img, cv2.COLOR_RGB2BGR))
    Generate_list('G:/programs/PSPNet-Keras-tensorflow-master/example_results_city_val')