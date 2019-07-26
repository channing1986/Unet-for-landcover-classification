__author__ = 'Changlin'
__version__ = 0.1

import os

GPUS = '0'  # GPU indices to restrict usage




DATA_FOLDERS={}
DATA_NAMES={}
DATA_FOLDERS[0]='G:/DataSet/BuildingFacade/CMP/CMP_facade_DB_base'
DATA_FOLDERS[1]='G:/DataSet/BuildingFacade/EcoleCentraleParisFacadesDatabase/monge_labelled'
DATA_NAMES[0]='CMP'
DATA_NAMES[1]='ParisFacades'

train_background=11
train_facade=0  
train_window=1
train_door=2 
train_cornice=3 
train_sill=4
train_balcony=5 
train_blind=6
train_deco=7 
train_molding=8 
train_pillar=9
train_shop=10

show_background=[0,0,255]
show_facade=[0,255,255]  
show_window=[0,85,255]
show_door=[0,170,255] 
show_cornice=[255,170,0] 
show_sill=[85,255,170]
show_balcony=[170,255,85] 
show_blind=[255,255,0]
show_deco=[255,85,0]
show_molding=[0,0,170] 
show_pillar=[255,0,0]
show_shop=[170,0,0]


TRAIN2SHOW={}
TRAIN2SHOW[train_facade]=show_facade
TRAIN2SHOW[train_window]=show_window
TRAIN2SHOW[train_door]=show_door
TRAIN2SHOW[train_shop]=show_shop
TRAIN2SHOW[train_cornice]=show_cornice
TRAIN2SHOW[train_sill]=show_sill
TRAIN2SHOW[train_balcony]=show_balcony
TRAIN2SHOW[train_blind]=show_blind
TRAIN2SHOW[train_deco]=show_deco
TRAIN2SHOW[train_molding]=show_molding
TRAIN2SHOW[train_pillar]=show_pillar
TRAIN2SHOW[train_background]=show_background

SHOW2TRAIN={}
#TRAIN2SHOW[show_facade]=train_facade
#TRAIN2SHOW[show_window]=train_window
#TRAIN2SHOW[show_door]=train_door
#TRAIN2SHOW[show_shop]=train_shop
#TRAIN2SHOW[show_cornice]=train_cornice
#TRAIN2SHOW[show_sill]=train_sill
#TRAIN2SHOW[show_balcony]=train_balcony
#TRAIN2SHOW[show_blind]=train_blind
#TRAIN2SHOW[show_deco]=train_deco
#TRAIN2SHOW[show_molding]=train_molding
#TRAIN2SHOW[show_pillar]=train_pillar
#TRAIN2SHOW[show_background]=train_background