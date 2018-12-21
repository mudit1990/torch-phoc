"""
This is an older version. Mainly for experiments. Use extend_boundary.py
"""
import numpy as np
from scipy.misc import imsave
from PIL import Image
from scipy.ndimage import imread
import math

#fN = 'D0006-0285025'
#fN = 'D0117-5755025'
#fN = 'D0117-5755024'
#fN = 'D0117-5755018'
fN = 'D0090-5242001'

im_fN='../detection_outputs/'+fN+'.npy' #90 has 536 clips
align_fN='../detection_alignment/'+fN+'.npy' #90 has 536 entries
annot_fN='../jerods_annotations/'+fN+'.npy' #90 has 604 keys in the dictionary

images=np.load(im_fN)
aligns=np.load(align_fN)
annots=np.load(annot_fN).item()

def extend(pt1,pt2,change,direct):
    if pt1[0]!=pt2[0]:
        m = 1.0*(pt2[1]-pt1[1])/(pt2[0]-pt1[0])
        x_ch = change * 1.0/(math.sqrt(1+m*m))
        y_ch = change * m/(math.sqrt(1+m*m))
    else:
        x_ch = 0
        y_ch = change
    if direct == -1: #Right
        x3 = pt2[0]+x_ch
        y3 = pt2[1]+y_ch
    else:
        x3 = pt1[0]-x_ch
        y3 = pt1[1]-y_ch
    return [x3,y3]

def distance(x1, x2):
    return (math.hypot(x2[0] - x1[0], x2[1] - x1[1]))

def orientation(x1, x2):
    if float(x2[0] - x1[0]) == 0:
        if x2[1] - x1[1] > 0:
            return -90
        else:
            return 90
    else:
        return math.degrees(math.atan2(x2[1] - x1[1], x2[0] - x1[0]))

def create_new_outputs2(output_path='../detection_outputs_new/',in_fN='../images_to_extend/image_dir_'):
    """
    THis func. calculates and saves the new detection outputs 
    """
    '''
    f = open(fN, 'rb')
    word_lst=[]
    direct_lst=[]
    i=0
    for line in f:
        if len(line)<2;
            break
        line=line.replace('\n')
        line_lst=line.split(',')
        word_lst.append(str(line_lst[0]))
        direct_lst.append(int(line_lst[1]))
        i+=1
    f.close()
    '''
    in_fN+=(fN+'.npy')
    output_path+=fN+'.npy'
    print("Doing for image: "+fN)
    arr = np.load(in_fN)
    word_lst = list(arr[2])
    direct_lst = list(arr[1])
    indices_lst = list(arr[0])
    new_coords=[]
    print("Read lines from Mudit's file")
    for i in range(len(word_lst)):
        ori_coords = list(images[indices_lst[i]])
        saved_coords = list(ori_coords)
        #ori_coords = [ori_coords[3][0],ori_coords[3][1],ori_coords[1][0],ori_coords[1][1]]
        #Instead of directly choosing 2 and 1. Pick longer dimension
        #length_im = ori_coords[2]-ori_coords[0]
        width = distance(ori_coords[0], ori_coords[1]) #dim_1
        height = distance(ori_coords[0],ori_coords[3]) #dim_2
        #length_wo = len(word_lst[i])
        length_wo = word_lst[i]
        if width>height:
            length_im = width
            if direct_lst[i]==1: #Increase left
                #ori_coords[0]-=int(length_im/length_wo)
                #saved_coords[0][0]-= int(length_im/length_wo)
                #saved_coords[3][0]-= int(length_im/length_wo)
                new_pt1 = extend(ori_coords[0],ori_coords[1],length_wo,1)
                new_pt2 = extend(ori_coords[3],ori_coords[2],length_wo,1)
                saved_coords = [new_pt1,ori_coords[1],ori_coords[2],new_pt2]
            elif direct_lst[i]==-1: #Increase to right
                #ori_coords[2]+=int(length_im/length_wo)
                #saved_coords[2][0]+= int(length_im/length_wo)
                #saved_coords[1][0]+= int(length_im/length_wo)
                new_pt1 = extend(ori_coords[0],ori_coords[1],length_wo,-1)
                new_pt2 = extend(ori_coords[3],ori_coords[2],length_wo,-1)
                saved_coords = [ori_coords[0],new_pt1,new_pt2,ori_coords[3]]
            elif direct_lst[i]==0: #Don't
                ori_coords[2]*=1 #-_-
        else:
            length_im = height
            if direct_lst[i]==1: #Increase top basically
                #ori_coords[1]-=int(length_im/length_wo)
                #saved_coords[2][1]-= int(length_im/length_wo)
                #saved_coords[3][1]-= int(length_im/length_wo)
                new_pt1 = extend(ori_coords[0],ori_coords[3],length_wo,-1) #-1!
                new_pt2 = extend(ori_coords[1],ori_coords[2],length_wo,-1)
                saved_coords = [ori_coords[0],ori_coords[1],new_pt2,new_pt1]
            elif direct_lst[i]==-1: #Increase below basically
                #ori_coords[3]+=int(length_im/length_wo)
                #saved_coords[0][1]+= int(length_im/length_wo)
                #saved_coords[1][1]+= int(length_im/length_wo)
                new_pt1 = extend(ori_coords[0],ori_coords[3],length_wo,1) #1!
                new_pt2 = extend(ori_coords[1],ori_coords[2],length_wo,1)
                saved_coords = [new_pt1,new_pt2,ori_coords[2],ori_coords[3]]
            elif direct_lst[i]==0: #Don't
                ori_coords[3]*=1 #-_-
        #new_coords.append([[ori_coords[0],ori_coords[3]],[ori_coords[2],ori_coords[3]],[ori_coords[2],ori_coords[1]],[ori_coords[0],ori_coords[1]]])
        new_coords.append([[saved_coords[0][0],saved_coords[0][1]],[saved_coords[1][0],saved_coords[1][1]],[saved_coords[2][0],saved_coords[2][1]],[saved_coords[3][0],saved_coords[3][1]]])
        #new_coords.append([ori_coords[0],ori_coords[1],ori_coords[2],ori_coords[3]])
        if i%100==0:
            print(i,len(word_lst))        
    np.save(output_path,new_coords)
    print("Saved at: "+output_path)

def test_cropping(im_path,im_id,direct,word,new_path,before_path=None,ignore_top=False):
    #For debug purposes. You can save original crop and new_clip
    #image = imread('image_0.png') #(135,487,3). Olathe
    #imsave("../rm_im_3.png",image[0:100,0:400,:]) #Olath    
    ori_coords = list(np.copy(images[im_id])) #[array([2529,1271]),array([2671,1271]),array([2671,1242]),array([2529,1242])]
    saved_coords = list(ori_coords) #copy of ori_coords
    ori_coords = [ori_coords[3][0],ori_coords[3][1],ori_coords[1][0],ori_coords[1][1]]
    image_obj = Image.open(im_path+'.jpg')
    if before_path!=None:
        cropped_im = image_obj.crop(ori_coords) #(2529,1242,2671,1271)
        cropped_im.save(before_path)
        print("Saved at: "+str(before_path))
        print("Note that this is NOT exactly the original clip (rectangle switchroo)")
        print(saved_coords)
    len_word=len(word)
    dim_1 = ori_coords[2]-ori_coords[0]
    dim_2 = ori_coords[3]-ori_coords[1]
    if dim_1>=dim_2 or ignore_top==True: #Not exactly a correct one
        length = ori_coords[2]-ori_coords[0]
        print(length,len_word,int(length/len_word))
        if direct==1: #Increase left. 2529 is decreasing
            ori_coords[0]-=int(length/len_word) #If u want more letters, multiply here directly
            saved_coords[0][0]-= int(length/len_word)
            saved_coords[3][0]-= int(length/len_word)
        elif direct==-1: #Right. 2671 is increasing
            ori_coords[2]+=int(length/len_word)
            saved_coords[2][0]+= int(length/len_word)
            saved_coords[1][0]+= int(length/len_word)
    else: #Can increase top and bottom also
        length = dim_2
        print(length,len_word,int(length/len_word))
        if direct==1: #Increase top. 1242 is decreasing
            ori_coords[1]-=int(length/len_word)
            saved_coords[2][1]-= int(length/len_word)
            saved_coords[3][1]-= int(length/len_word)
        elif direct==-1: #Bottom. 1271 is increasing
            ori_coords[3]+=int(length/len_word)
            saved_coords[0][1]+= int(length/len_word)
            saved_coords[1][1]+= int(length/len_word)
    cropped_im = image_obj.crop(ori_coords) #(2480,1242,2671,1271)
    cropped_im.save(new_path)
    print("Saved at: "+new_path)
    print("Note that this is NOT exactly the original clip (rectangle switchroo)")
    print([[saved_coords[0][0],saved_coords[0][1]],[saved_coords[1][0],saved_coords[1][1]],[saved_coords[2][0],saved_coords[2][1]],[saved_coords[3][0],saved_coords[3][1]]])
'''
im_id=0
word = annots[aligns[im_id][0]-1]['name'] #Cheating!!!!! You are seeing ground truth
print(word,len(word))
direct=-1
if direct==1:
    new_path=fN[:5]+'_'+str(im_id)+"_left"+'.jpg'
elif direct==-1:
    new_path=fN[:5]+'_'+str(im_id)+"_right"+'.jpg'
test_cropping(im_path='D0090-5242001',im_id=im_id,direct=direct,word=word,new_path=new_path)
'''
create_new_outputs2()
#test_cropping("D0006-0285025",98,1,"Paris","rm_paris_left.png","rm_paris_original.png")
#test_cropping("D0090-5242001",314,-1,"Neosho","rm_neosho_bottom.png","rm_neosho_original.png")
#test_cropping("D0090-5242001",314,1,"Neosho","rm_neosho_top.png","rm_neosho_original.png")
#test_cropping("D0090-5242001",314,-1,"Neosho","rm_neosho_right.png","rm_neosho_original.png",ignore_top=True)
#test_cropping("D0090-5242001",314,1,"Neosho","rm_neosho_left.png","rm_neosho_original.png",ignore_top=True)

