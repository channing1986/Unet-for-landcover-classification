__author__ = 'Changlin'
__version__ = 0.1

import os

GPUS = '2'  # GPU indices to restrict usage



CHECKPOINT_DIR = './checkpoint_track3_single_psp_bridge/'
MODEL_SAVE_PERIOD=10
BACKBONE = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'



DATA_FOLDERS={}
DATA_NAMES={}
DATA_FOLDERS[0]='G:/DataSet/BuildingFacade/CMP/CMP_facade_DB_base'
DATA_FOLDERS[1]='G:/DataSet/BuildingFacade/EcoleCentraleParisFacadesDatabase/monge_labelled'
DATA_NAMES[0]='CMP'
DATA_NAMES[1]='ParisFacades'
Num_Category=11
NUM_CATEGORIES=11
NUM_CHANNELS=3

category_names={}
category_names[0]='facade'
category_names[1]='window'
category_names[2]='door'
category_names[3]='cornice'
category_names[4]='sill'
category_names[5]='balcony'
category_names[6]='blind'
category_names[7]='deco'
category_names[8]='pillar'
category_names[9]='shop'
category_names[10]='background'
category_names[11]='molding'


train_facade=0  
train_window=1
train_door=2 
train_cornice=3 
train_sill=4
train_balcony=5 
train_blind=6
train_deco=7 
train_pillar=8
train_shop=9
train_background=10
train_molding=11


show_background=[0,0,255]
show_molding=[0,0,170] 
show_facade=[0,255,255]  
show_window=[0,85,255]
show_door=[0,170,255] 
show_cornice=[255,170,0] 
show_sill=[85,255,170]
show_balcony=[170,255,85] 
show_blind=[255,255,0]
show_deco=[255,85,0]
show_pillar=[255,0,0]
show_shop=[170,0,0]

TRAIN2SHOW={}
TRAIN2SHOW[train_facade]=show_facade
TRAIN2SHOW[train_window]=show_window
TRAIN2SHOW[train_door]=show_door
TRAIN2SHOW[train_cornice]=show_cornice
TRAIN2SHOW[train_sill]=show_sill
TRAIN2SHOW[train_balcony]=show_balcony
TRAIN2SHOW[train_blind]=show_blind
TRAIN2SHOW[train_deco]=show_deco
TRAIN2SHOW[train_molding]=show_molding
TRAIN2SHOW[train_pillar]=show_pillar
TRAIN2SHOW[train_shop]=show_shop
TRAIN2SHOW[train_background]=show_background

SHOW2TRAIN={}
SHOW2TRAIN[0]=show_facade
SHOW2TRAIN[1]=show_window
SHOW2TRAIN[2]=train_door
SHOW2TRAIN[3]=train_shop
SHOW2TRAIN[4]=train_cornice
SHOW2TRAIN[5]=train_sill
SHOW2TRAIN[6]=train_balcony
SHOW2TRAIN[7]=train_blind
SHOW2TRAIN[8]=train_deco
SHOW2TRAIN[10]=train_molding
SHOW2TRAIN[9]=train_pillar
SHOW2TRAIN[11]=train_background