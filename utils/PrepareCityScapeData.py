def get_data_list(file):

    img_folder='G:/DataSet/VOC2012/JPEGImages/'
    lab_folder='G:/DataSet/VOC2012/SegmentationClass/'
    data_suffix='.jpg'
    label_suffix='.png'
    fp = open(file)
    lines = fp.readlines()
    fp.close()

    data_files=[]
    label_files=[]
    nb_sample = len(lines)
    for line in lines:
            line = line.strip('\n')
            data_files.append(img_folder+line + data_suffix)
            label_files.append(lab_folder+line + label_suffix)
    return data_files,label_files

def input_generator_mp(img_pathes,label_pathes,batch_size):
    """

    """
    import multiprocessing as mp
    pool=mp.Pool();
    
    N = len(img_pathes) #total number of images

    idx = np.random.permutation(N) #shuffle the order

    batchInds = get_batch_inds(batch_size, idx, N) # generate each batch's idx like this:{[10,23,....],[1,1232,44,...],[....]....} 

            
    while True:
        for inds in batchInds:
            imgdata=[]
            labels=[]
            img_batch = [img_pathes[ind] for ind in inds]
            label_batch = [label_pathes[ind] for ind in inds]
            #aa=load_img_label(img_batch[0],label_batch[0])######
            img_label=zip(img_batch,label_batch)
            res=pool.starmap(load_img_label,img_label)
            for img,lable in res:
                imgdata.append(img)
                labels.append(lable)
            labels = np.array(labels, np.float32)
            imgdata = np.array(imgdata, np.float32)/255.
    #return imgdata,labels
            yield (imgdata, labels)
def load_img_label(rgb_file,gts_file):
    from PIL import Image
   
    #rgb_file = inputDict['rgb']
    #gts_file = inputDict['gts']
    rgb = cv2.imread(rgb_file)
 #   rgbh = image.img_to_array(rgbh)
    gts = cv2.imread(gts_file,0)

    gts = Image.open(gts_file )
    gts = img_to_array(gts.resize((320, 320), Image.NEAREST),data_format='channels_last').astype(int)
    #gts=np.squeeze(gts)
    #np.savetxt('gts1.txt', gts)
    gts[np.where(gts == 255)] = 21
    #np.savetxt('gts2.txt', gts)
    #gts=np.squeeze(gts)
    rgb=cv2.resize(rgb,(320,320))

    #gts=cv2.resize(gts,(320,320),interpolation=cv2.INTER_NEAREST)

    #y = np.zeros((gts.shape[0], gts.shape[1],2), dtype=np.float32)
    #for i in range(gts.shape[0]):
    #    for j in range(gts.shape[1]):

    #        if gts[i][j]<1:     ## for 2 channels output
    #            cc=0
    #        else:
    #            cc=1
    #        y[i,j,cc]=1


    return rgb, gts

def convert_image():
    img_train, lable_train=get_data_list(text_folder+'train.txt')


def main(cityscapesPath='', output_folder='', data_lable='data'):
    # Where to look for Cityscapes
    cityscapesPath='G:/DataSet/CityScapes/cityscapes'
    output_folder='G:/DataSet/CityScapes_cl/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 
    # how to search for all ground truth
    if data_lable=='data':
        searchFine   = os.path.join( cityscapesPath , "leftImg8bit"   , "*" , "*" , "*__leftImg8bit.png" )
    elif(data_lable=='label'):
        searchFine   = os.path.join( cityscapesPath , "gtFine"   , "*" , "*" , "*_gtFine_labelTrainIds.png" )
    else:
        printError( "Please input the right data_lable,which input is {}.".format(data_lable) )
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
        printError( "Did not find any files. Please consult the README." )

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        # create the output filename
        dst = f.replace( "_polygons.json" , "_labelTrainIds.png" )

        # do the conversion
        try:
            json2labelImg( f , dst , "trainIds" )
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()

# call the main
if __name__ == "__main__":
    main()