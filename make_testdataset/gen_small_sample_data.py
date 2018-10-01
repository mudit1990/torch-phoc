import glob
from PIL import Image, ImageFont, ImageDraw
import os
import fnmatch
import matplotlib
import random
import math
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg
import numpy as np
import re
import cv2
from tqdm import tqdm
import itertools
from scipy import ndimage
from operator import sub
'''
fonts_list = []
for root, dirnames, filenames in os.walk('../fonts_new/'):
    for filename in fnmatch.filter(filenames, '*.ttf'):
        fonts_list.append(os.path.join(root, filename))

background_images = []
for i in range(1, 6):
    my_file = Path('../map_textures/map_crop_0' + str(i) + '.jpg')
    if my_file.is_file():
        img = mpimg.imread('../map_textures/map_crop_0' + str(i) + '.jpg')
        background_images.append(img)
'''
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
    imgR = cv2.resize( imgR, (0,0), fx=ratio, fy=ratio )
    return imgR

def get_word_image(word, fonts_list):
    W, H = (227,227)
    #word = word_list[random.randint(0,len(word_list)-1)]
    #word = changeCase(word)
    font_name = fonts_list[0]
    fontsize = 2
    if len(word) <= 2:
        img_fraction = 0.5
    else:
        img_fraction = 0.75
    font = ImageFont.truetype(font_name, fontsize)
    image = Image.new("L",(W,H),"white")
    while font.getsize(word)[0] < img_fraction*image.size[0]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype(font_name, fontsize)

    draw = ImageDraw.Draw(image)
    w, h = font.getsize(word)
    draw.text((W/2 - w/2,H/2-h/2), word, font=font, fill="black")
    #image = image.crop((0,10,w+20,h+20))
    return image

def get_random_crop(size, background_images):
    image_number = random.randint(0, len(background_images)-1)
    img = background_images[image_number]
    # print(img.shape)
    height, width = img.shape[0], img.shape[1]
    start_row = random.randint(0, height - size[0]*2)
    start_column = random.randint(0, width - size[1]*2)
    new_img = img[start_row:start_row + size[0]*2, start_column:start_column + size[1]*2, :]
    new_img = new_img[..., list(list(itertools.permutations([0, 1, 2]))[random.randint(0, 5)])]
    result = cv2.resize(new_img,(size[1],size[0]))
    #print result.shape
    return result

def transform_image(img):
    W, H = (512,512)
    #img = cv2.resize(img, (W, H))
    width,height = (512,512)
    rotateFlag = random.randint(0, 1)
    if rotateFlag:
        rotateAngle = random.randint(-10,10)
        M = cv2.getRotationMatrix2D((width / 2, height / 2), rotateAngle, 1)
        img = cv2.warpAffine(img, M, (width, height), borderValue=(255, 255, 255))
    affineFlag = random.randint(0, 1)
    if affineFlag:
        pts1 = np.float32([[10, 10], [200, 50], [50, 200]])
        pts2 = np.float32([[10 + random.randint(-20, 20), 30 + random.randint(-20, 20)]
                              , [200, 50],
                           [50 + random.randint(-20, 20), 200 + random.randint(-20, 20)]])

        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (width, height), borderValue=(255, 255, 255))
    #print img.shape
    min_row = H
    max_row = 0
    min_col = W
    max_col = 0
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    for i in range(thresh1.shape[0]):
        for j in range(thresh1.shape[1]):
            if thresh1[i][j]==0:
                if i<min_row:
                    min_row = i
                if i>max_row:
                    max_row = i
                if j < min_col:
                    min_col = j
                if j > max_col:
                    max_col = j
    
    thresh1 = thresh1[min_row-10:max_row+10, min_col-10:max_col+10]
    
    return thresh1

def merge_background_text(img, bg_image):
    (r,g,b) = (random.randint(0,20),random.randint(0,20),random.randint(0,20))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 255:
                if random.random()<0.98:
                    bg_image[i, j, 0] = r
                    bg_image[i, j, 1] = g
                    bg_image[i, j, 2] = b
    kernel = np.ones((3,3),np.float32)/9
    bg_image = cv2.filter2D(bg_image,-1,kernel)
    #     bg_image = cv2.resize(bg_image, (227,227))
    return bg_image

def pad_image(img):
    (W,H) = (227, 227)
    (h, w) = (img.shape[0], img.shape[1])
    constant= cv2.copyMakeBorder(img,(H-h)/2, (H - (h+(H-h)/2)),(W-w)/2,(W - (w+(W-w)/2)),cv2.BORDER_CONSTANT,value=(255,255,255))
    return constant


def get_color_text_image(word):
    '''
    Generate color image from a background
    '''
    try:
        img = np.array(get_word_image(word))
        while True:
            temp_img = np.copy(img)
            temp_img = transform_image(temp_img)
            print temp_img.shape
            if temp_img.shape[0]!=0 and temp_img.shape[1]!=0:
                img = temp_img
                break
        bg_image = get_random_crop(img.shape)
        result = merge_background_text(img, bg_image)
        padded_result = pad_image(result)
    except Exception as e:
        print e
        return (None,False)
    return padded_result,True

def generate_left_words(list_of_files, path_to_images, path_to_anots, save_dir):
    '''
    This function just copies the rectangular words
    from the map images
    save in a directory
    '''
    # A is a dictionary of dictionaries
    A = {}
    for i in range(len(list_of_files)):
        _dict = np.load(path_to_anots+str(list_of_files[i])+'.npy').item()
        for j in _dict.keys():
            if len(_dict[j]['vertices']) != 4:
                del _dict[j]
        A[i] = _dict

    # dictionary_of_indices is dict of indices of each dicts 
    dictionary_of_indices = {}
    for i in range(len(A)):
        dictionary_of_indices[i] = A[i].keys()

    # read the images in a dic too
    I = {}
    for i in range(len(list_of_files)):
        I[i] = mpimg.imread(path_to_images+str(list_of_files[i])+'.tiff')

    list_of_words = []
    y = []
    filenames = []
    for count in range(0,10000):
        print 'image %d' %count
        # randomly pick a file and a rectangle
        file_ID = np.random.randint(len(A))
        loops = 0
        while len(dictionary_of_indices[file_ID]) <= 1:
            file_ID = np.random.randint(len(A))
            loops = loops+1
            if loops == 20:
                return list_of_words, y, filenames
        anots_ID = np.random.choice(dictionary_of_indices[file_ID])

        print file_ID, anots_ID
        # remove the rectangle from the available keys
        dictionary_of_indices[file_ID].remove(anots_ID)
    
        # now get the information from the dictionary
        image_from_map_info = A[file_ID][anots_ID]
    
        # fulcrum or pivot for map's rotation
        fulcrum = map(int,image_from_map_info['vertices'][0])
        x2 = image_from_map_info['vertices'][1]
        x4 = image_from_map_info['vertices'][3]
        width = int(distance(fulcrum, x2))
        height = int(distance(fulcrum, x4))
        _angle = orientation(fulcrum, x2)
    
        I_cache = np.copy(I[file_ID])
        I_cache, fulcrum = get_crop(I_cache, image_from_map_info['vertices'], fulcrum)
        #get the final crop
        extracted_crop = rotateImage(I_cache, _angle, fulcrum, height, width)

        # get padded image
        final_img = pad_image(extracted_crop)

        true_label = np.random.randint(0,1)
        if true_label == 0:
            anots_ID = np.random.choice(dictionary_of_indices[file_ID])
            label = A[file_ID][anots_ID]['name']
        else:
            label = image_from_map_info['name']
    
        list_of_words.append(label)
        y.append(true_label)
        
        # save the image
        filenames.append(save_dir+str(count))
        cv2.imwrite(save_dir+str(count)+'.png', final_img)

    return list_of_words, y, filenames

def generate_right_words(list_of_words, save_dir):
    '''
    Generates word images from associated annotations
    save in a directory
    '''
    filenames = []
    count = 0
    for word in list_of_words:
        print 'word %d' %count
        img,flag = get_color_text_image(word)
        # save the image
        filenames.append(save_dir+str(count))
        cv2.imwrite(save_dir+str(count)+'.png', img)
        count = count+1
    return filenames

def get_color_text_image_true(word, fonts_list, background_images, padded=True, bg=True):
    '''
    Generate color image from a background
    '''
    print bg
    img = np.array(get_word_image(word, fonts_list))
    temp_img = np.copy(img)
    temp_img = transform_image(temp_img)
    if temp_img.shape[0]!=0 and temp_img.shape[1]!=0:
        img = temp_img
    bg_image = get_random_crop(img.shape, background_images)
    if bg:
        result = merge_background_text(img, bg_image)
    else:
        result = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    if padded:
        padded_result = pad_image(result)
    else:
        padded_result = cv2.resize(result, dsize=(487, 135), interpolation=cv2.INTER_CUBIC)
    return padded_result


def generate_word_images_from_list(list_of_words, fonts_list, background_images, padded=True, bg=True):
    '''
    Generates word images from associated annotations
    save in a directory
    '''
    images = []
    count = 0
    for word in list_of_words:
        print 'word '+word
        img = get_color_text_image_true(word, fonts_list, background_images, padded, bg)
        images.append(img)
    return images

def generate_left_words_from_image(list_of_files, path_to_images, path_to_anots, padded=True, aspect=False):
    '''
    This function just copies the rectangular words
    from the map images
    save in a directory
    '''
    # A is a dictionary of dictionaries
    print padded
    A = {}
    for i in range(len(list_of_files)):
        _dict = np.load(path_to_anots+str(list_of_files[i])+'.npy').item()
        for j in _dict.keys():
            if len(_dict[j]['vertices']) != 4:
                del _dict[j]
        A[i] = _dict

    # dictionary_of_indices is dict of indices of each dicts 
    dictionary_of_indices = {}
    for i in range(len(A)):
        dictionary_of_indices[i] = A[i].keys()

    # read the images in a dic too
    I = {}
    for i in range(len(list_of_files)):
        I[i] = mpimg.imread(path_to_images+str(list_of_files[i])+'.tiff')

    list_of_words = []
    y = []
    image_files = []
    for count in range(0,10000):
        print 'image %d' %count
        # randomly pick a file and a rectangle
        file_ID = np.random.randint(len(A))
        loops = 0
        while len(dictionary_of_indices[file_ID]) <= 1:
            file_ID = np.random.randint(len(A))
            loops = loops+1
            if loops == 20:
                return list_of_words, y, image_files
        anots_ID = np.random.choice(dictionary_of_indices[file_ID])

        print file_ID, anots_ID
        # remove the rectangle from the available keys
        dictionary_of_indices[file_ID].remove(anots_ID)
    
        # now get the information from the dictionary
        image_from_map_info = A[file_ID][anots_ID]
    
        # fulcrum or pivot for map's rotation
        fulcrum = map(int,image_from_map_info['vertices'][0])
        x2 = image_from_map_info['vertices'][1]
        x4 = image_from_map_info['vertices'][3]
        width = int(distance(fulcrum, x2))
        height = int(distance(fulcrum, x4))
        _angle = orientation(fulcrum, x2)
    
        I_cache = np.copy(I[file_ID])
        I_cache, fulcrum = get_crop(I_cache, image_from_map_info['vertices'], fulcrum)
        #get the final crop
        extracted_crop = rotateImage(I_cache, _angle, fulcrum, height, width)

        # get padded image
        if padded:
            final_img = pad_image(extracted_crop)
        else:
            if aspect:
                final_img = cv2.resize(extracted_crop, dsize=(487, 135), interpolation=cv2.INTER_CUBIC)
            else:
                final_img = extracted_crop

        true_label = 1#np.random.randint(0,1)
        if true_label == 0:
            anots_ID = np.random.choice(dictionary_of_indices[file_ID])
            label = A[file_ID][anots_ID]['name']
        else:
            label = image_from_map_info['name']
    
        list_of_words.append(label)
        y.append(true_label)
        
        # save the image
        image_files.append(final_img)
        #print(list_of_words[count])
        #plt.imshow(image_files[count])
        #plt.show()
        #filenames.append(save_dir+str(count))
        #cv2.imwrite(save_dir+str(count)+'.png', final_img)

    return list_of_words, y, image_files