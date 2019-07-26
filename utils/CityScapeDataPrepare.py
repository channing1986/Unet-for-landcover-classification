# python imports
from __future__ import print_function
import os, glob, sys
import cv2

# The main method
def prepare_patch_resized_data(re_size=[320,320],patch_size=[80,80], cityscapesPath='', output_folder='', data_type='train',img_label='img'):
    # Where to look for Cityscapes

    re_size=[256,512]
    cityscapesPath='G:/DataSet/CityScapes/cityscapes'
    output_folder_patch_='G:/DataSet/CityScapes_cl3/patch/'
    output_folder_resized_='G:/DataSet/CityScapes_cl3/resized/'
    output_folder_patch= os.path.join('G:/DataSet/CityScapes_cl3/patch/',img_label)
    output_folder_resized= os.path.join('G:/DataSet/CityScapes_cl3/resized/',img_label)
    if not os.path.exists(output_folder_patch):
        os.makedirs(output_folder_patch) 
    if not os.path.exists(output_folder_resized):
        os.makedirs(output_folder_resized)
    # how to search for all ground truth
    if img_label=='img':
        searchFine   = os.path.join( cityscapesPath , "leftImg8bit"   , data_type , "*" , "*_leftImg8bit.png" )
    elif(img_label=='label'):
        searchFine   = os.path.join( cityscapesPath , "gtFine"   , data_type , "*" , "*_gtFine_labelTrainIds.png" )
    elif(img_label=='eval_label'):
        searchFine   = os.path.join( cityscapesPath , "gtFine"   , data_type , "*" , "*_gtFine_labelIds.png" )

    else:
        print( "Please input the right img_label,which input is {}.".format(img_label) )
    #searchCoarse = os.path.join( cityscapesPath , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" ) darmstadt_000000_000019_gtFine_labelTrainIds.png

    # search files
    filesFine = glob.glob( searchFine )
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
    patch_list=[]
    resize_list=[]
    for f in files:
        # create the output filename
        if img_label=='img':
            rgb = cv2.imread(f)
        else:
            rgb = cv2.imread(f,0)
        ixx=f.rfind('\\')
        file_name=f[ixx+1:-4]
        ###############
        if img_label=='img':
            resized_rgb=cv2.resize(rgb,(re_size[1],re_size[0]))
        else:
            resized_rgb=cv2.resize(rgb,(re_size[1],re_size[0]),interpolation=cv2.INTER_NEAREST)

        resize_name=os.path.join(output_folder_resized,file_name+'.png') 
        cv2.imwrite(resize_name,resized_rgb)
        resize_list.append(resize_name)

        #########################

        list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(resized_rgb.shape[0] // patch_size[0])]
        list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(resized_rgb.shape[1] // patch_size[1])]

        list_row_idx_=list_row_idx[1:3]    ###only use the middel level image patches, not include the top and bottom part of the image.
        count=0
        for row_idx in list_row_idx_:
            for col_idx in list_col_idx:      
                count+=1
                if img_label=='img':

                    patch=resized_rgb[row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :]
                    patch=cv2.resize(patch,(re_size[1],re_size[0]))
                else:
                    patch=resized_rgb[row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]]
                    patch=cv2.resize(patch,(re_size[1],re_size[0]),interpolation=cv2.INTER_NEAREST)
                patch_name=os.path.join(output_folder_patch,file_name+'-'+str(count)+'.png')
                cv2.imwrite(patch_name,patch)
                patch_list.append(patch_name)
        

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()
    patch_text=output_folder_patch_+'//'+img_label+'_list.txt'
    f= open (patch_text,"w") 
    for name in patch_list:
            f.write(name)
            f.write('\n')
    f.close()

    resized_text=output_folder_resized_+'//'+img_label+'_list.txt'
    f= open (resized_text,"w") 
    for name in resize_list:
            f.write(name)
            f.write('\n')
    f.close()
# call the main
if __name__ == "__main__":
    #prepare_patch_resized_data(img_label='label')   #img  label
    prepare_patch_resized_data( data_type='val',img_label='eval_label')
