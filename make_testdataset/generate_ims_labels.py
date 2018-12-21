"""
Generates after extension regions and labels for input map name.
"""

import numpy as np
import glob
import matplotlib.image as mpimg
import math
from scipy import ndimage
import cv2
from tqdm import tqdm
import sys

def distance(x1, x2):
    return int(math.hypot(x2[0] - x1[0], x2[1] - x1[1]))

def orientation(x1, x2):
    if float(x2[0] - x1[0]) == 0:
        if x2[1] - x1[1] > 0:
            return -90
        else:
            return 90
    else:
        return math.degrees(math.atan2(x2[1] - x1[1], x2[0] - x1[0]))

def get_crop(Img, V, fulcrum):
    '''
    get a good crop of the region around/bounded by V
    '''
    V = np.asarray(V)
    rowmin = int(min(V[:,1]))
    rowmax = int(max(V[:,1]))
    colmin = int(min(V[:,0]))
    colmax = int(max(V[:,0]))
    Img_out = Img[rowmin:rowmax+1, colmin:colmax+1, :]
    fulcrum = np.asarray(fulcrum) - np.asarray([colmin, rowmin])
    return Img_out , fulcrum

def rotateImage(img, angle, pivot, height, width):
    '''
    rotate the image
    '''
    padX = [300+int(img.shape[1] - pivot[0]), 300+int(pivot[0])]
    padY = [300+int(img.shape[0] - pivot[1]), 300+int(pivot[1])]
    imgP = np.pad(img, [padY, padX, [0, 0]], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    centerRow = int(imgR.shape[0]/2)
    centerCol = int(imgR.shape[1]/2)
    imgR = imgR[centerRow-height+1:centerRow+1, centerCol : centerCol+width-1, :]
    # crop to a fixed size
    if imgR.shape[1] > imgR.shape[0]:
        ratio = float(188)/imgR.shape[1]
    else:
        ratio = float(188)/imgR.shape[0]
    try:
        imgR = cv2.resize( imgR, (0,0), fx=ratio, fy=ratio )
    except:
        print "In rotateImage() function"
    return imgR

def _save_data(_file_name, path_to_images, path_to_anots, path_to_detections, path_to_alignment,path_to_id):
    # A is a dictionary of dictionaries
    # B is list of all detections
    # C is the alignment matrix between B and C
    A = np.load(path_to_anots+_file_name+'.npy').item()
    B = np.load(path_to_detections+_file_name+'.npy')
    C = np.load(path_to_alignment+_file_name+'.npy')
    arr = np.load(path_to_id)
    arr = arr[0]
    idx = 0
    # get the image
    I = mpimg.imread(path_to_images+_file_name+'.tiff')

    _annotations = []
    _images = []
    detection_num=0 #Where it is failing
    # loop over all detections
    for i in tqdm(range(len(B))):
        fulcrum = map(int,B[i][0])
        x2 = B[i][1]
        x4 = B[i][3]
        width = int(distance(fulcrum, x2))
        height = int(distance(fulcrum, x4))
        _angle = orientation(fulcrum, x2)

        # get crop accordingly
        I_cache = np.copy(I)
        I_cache, fulcrum = get_crop(I_cache, B[i], fulcrum)
        extracted_crop = rotateImage(I_cache, _angle, fulcrum, height, width)
        try:
            final_img = cv2.resize(extracted_crop, dsize=(487, 135), interpolation=cv2.INTER_CUBIC) #This throws an error if im too small. 722,727,1278 in Mudit's 117 image_to_extend
        except Exception as e:
            print e
            #print "In other function"
            print str(detection_num)
            print str(B[i])
            continue #I"M CONTINUING. CAREFUL!
            
        # get label
        if C[arr[idx]] == 0:
            label = 'no label attached'
        else:
            try:
                label = str(A[int(C[arr[idx]] - 1)]['name'])
            except Exception as e:
                print(e)
                label = "no label attached"
        idx+=1
        _images.append(final_img)
        _annotations.append(label)
        detection_num+=1
    return _images, _annotations


path_to_images = '../../phoc_data/maps/'
path_to_anots = '../jerods_annotations/'
#path_to_detections = '../detection_outputs_new/ray_output_gis/' #Generated from extend_image.py
path_to_detections = '../detection_outputs_new/ray_output_normal/' # From extend_image.py
path_to_alignment = '../detection_alignment_lines/' # For ray_regions (and now extendby2)
#path_to_alignment = '../detection_alignment_lines_ext_2/'
#path_to_id = '../images_to_extend/image_dir_D0090-5242001.npy'
'''
image_lst=[]
for files in glob.glob('../detection_alignment/*'):
        _,_, align_file = files.rpartition('/')
        image_name, _, _ = align_file.rpartition('.')
        image_lst.append(image_name)
'''
#f = open('../splits/val_files.txt', 'rb')
#image_lst = f.readlines()
#f.close()
#image_lst = [x.rstrip('\n') for x in image_lst]
        
image_name = sys.argv[1]
print image_name
#path_to_id = '../images_to_extend/image_dir_'+image_name+'.npy'
#path_to_id = '../images_to_extend/ray_output_gis/image_dir_'+image_name+'.npy'
#path_to_id = '../images_to_extend/ray_output_ext_2/image_dir_'+image_name+'.npy'
path_to_id = '../images_to_extend/ray_output_normal/image_dir_'+image_name+'.npy'
original_images, original_words = _save_data(image_name, \
        path_to_images, path_to_anots, path_to_detections, path_to_alignment,path_to_id)
np.save('../detection_outputs_ready_for_test/ray_regions_normal_after/'+image_name+'.npy', original_images)
np.save('../detection_outputs_ready_for_test/ray_labels_normal_after/'+image_name+'.npy', original_words)
print(image_name+" done.")
