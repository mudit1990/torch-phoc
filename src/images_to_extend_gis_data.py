import argparse
import logging

import numpy as np
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

import copy
from datasets.maps_alt import MAPSDataset

#from cnn_ws.transformations.homography_augmentation import HomographyAugmentation
from cnn_ws.losses.cosine_loss import CosineLoss

from cnn_ws.models.myphocnet import PHOCNet
from cnn_ws.evaluation.retrieval import map_from_feature_matrix, map_from_query_test_feature_matrices
from torch.utils.data.dataloader import _DataLoaderIter as DataLoaderIter
from torch.utils.data.sampler import WeightedRandomSampler

from cnn_ws.utils.save_load import my_torch_save, my_torch_load

word_filter_len = 1 # only words above this length are considered valid
max_var_len = 2 # max length word variation that needs to be done

if not torch.cuda.is_available():
    logger.warning('Could not find CUDA environment, using CPU mode')
    gpu_id = None
else:
    gpu_id = [0]
#torch.cuda.get_device_name(gpu_id[0])
pass


model_ = torch.load('models/PHOCNet_Nov13.pt')
cnn = model_.module#list(model_.named_parameters())
if gpu_id is not None:
        if len(gpu_id) > 1:
            cnn = nn.DataParallel(cnn, device_ids=gpu_id)
            cnn.cuda()
        else:
            cnn.cuda(gpu_id[0])
cnn.training = False


from strlocale import BasicLocale

def clean_words(words):
    lc = BasicLocale()
    for i, w in enumerate(words):
        try:
            words[i] = lc.represent(w).encode('ascii',errors='ignore')
        except:
            words[i] = w
    return words

def load_and_transform(map_name):
    images = np.load('/mnt/nfs/work1/696ds-s18/mbhargava/detection_outputs_ready_for_test/ray_regions/org_clips/'+map_name+'.npy')
    words = np.load('/mnt/nfs/work1/696ds-s18/mbhargava/detection_outputs_ready_for_test/ray_labels/org_clips/'+map_name+'.npy')
    images = np.transpose(images, (0,3,1,2))
    words = clean_words(words)
    print 'Images Shape ', images.shape
    print 'Words Shape ', words.shape
    return images, words

def load_and_clean_gis_data():
    with open('../../../GIS_data/GIS_combined.txt') as f:
        gis_data = np.array(f.read().splitlines())
    gis_data = clean_words(gis_data)
    print 'GIS Data', gis_data.shape
    return gis_data


def gen_img_phoc_embs(cnn, images):
    outputs = []
    for i in tqdm(range(len(images)), ascii=True, desc='Converting Images to Embeddings'):
        word_img = images[i]
        word_img = 1 - word_img.astype(np.float32) / 255.0
        word_img = word_img.reshape((1,) + word_img.shape)
        word_img = torch.from_numpy(word_img).float()
        word_img = word_img.cuda(gpu_id[0])
        word_img = torch.autograd.Variable(word_img)
        output = torch.sigmoid(cnn(word_img))
        output = output.data.cpu().numpy().flatten()
        outputs.append(output)
    return outputs


# In[28]:


from cnn_ws.string_embeddings.phoc import build_phoc_descriptor

def insert_dict_set(dct, key, val):
    if key not in dct:
        dct[key] = set()
    dct[key].add(val)
    
# the method defines the rules to handle multiple dir associated with a given word
# returns conf_words which is a set of word_var where this confusion exists
# word_var: dictionary from word -> chosen_dir. Incase a word has multiple dir
# the following preference order is followed 0 > (1,-1) > (2,-2) > (3,-3) ...
def handle_word_conf(comp_word_var):
    word_var = {}
    conf_words = set()
    for var in comp_word_var.keys():
        dirs = np.array(list(comp_word_var[var]))
        if(len(dirs) == 1):
            word_var[var] = dirs[0]
        else:
            conf_words.add(var)
            idx = np.argmin(np.abs(dirs))
            word_var[var] = dirs[idx]
    return word_var, conf_words

# function to create word variations
# word_var is a dictionary that contains all variations as key and either 0 or 1 or -1 or 2 or -2 etc as value
# 0 denotes the root word, -1 denotes var = root_word[:-1], +1 denotes var = root_word[1:]
# root_word_var is a dict that stores original_word => all_variations
# enable_conf: boolean flag that controls if the confusion logic should be used.
# when enabled if a word has multiple directions associated with it (happens if root words ar rand and grand)
# it marks it as to be extended and also stores it in the confusion list
def create_word_variations(words, enable_conf=False):
    word_var = {}
    root_word_var = {}
    # create the root word variation dict and set word_var as -1 or +1
    for w in words:
        root_var_list = [w, w.lower(), w.upper(), w.capitalize()]
        var_set = set()
        for var in root_var_list:
            for l in range(1,max_var_len+1):
                if len(w) <= l:
                    continue
                insert_dict_set(word_var, var, 0)
                insert_dict_set(word_var, var[l:], l)
                insert_dict_set(word_var, var[:-l], -l)
                var_set.add(var)
                var_set.add(var[l:])
                var_set.add(var[:-l])
        root_word_var[w] = var_set
    word_var, conf_words = handle_word_conf(word_var)
    return word_var, root_word_var, conf_words

def gen_text_phoc_embs(words):
    word_strings = words
    unigrams = [chr(i) for i in range(ord('&'), ord('&')+1) + range(ord('A'), ord('Z')+1) +                     range(ord('a'), ord('z') + 1) + range(ord('0'), ord('9') + 1)]
    bigram_levels = None
    bigrams = None
    phoc_unigram_levels=(1, 2, 4, 8)
    
    word_var_dir, root_word_var, conf_words = create_word_variations(word_strings, enable_conf=True)
    
    embedding = build_phoc_descriptor(words=word_strings,
                                  phoc_unigrams=unigrams,
                                  bigram_levels=bigram_levels,
                                  phoc_bigrams=bigrams,
                                  unigram_levels=phoc_unigram_levels)

    word_var_strings = word_var_dir.keys()
    embedding_var = build_phoc_descriptor(words=word_var_strings,
                                  phoc_unigrams=unigrams,
                                  bigram_levels=bigram_levels,
                                  phoc_bigrams=bigrams,
                                  unigram_levels=phoc_unigram_levels)
    
    return (embedding, embedding_var, word_var_strings, word_var_dir, root_word_var, conf_words)
    


# In[29]:


# the new report matches method that handles variations
from scipy.spatial.distance import cdist, pdist, squareform
def report_matches_with_variations(outputs, embedding, matching, ground_truth, 
                                   words, word_var_dir, root_word_var, k, length):
    # length sorting stuff
    qualified_ids = [x for x in range(len(ground_truth)) if len(ground_truth[x]) > length]
    outputs = np.array(outputs)
    ground_truth = np.array(ground_truth)
    outputs = list(outputs[qualified_ids])
    ground_truth = list(ground_truth[qualified_ids])
    
    # the real computation
    dist_mat = cdist(XA=outputs, XB=embedding, metric=matching)
    retrieval_indices = np.argsort(dist_mat, axis=1)
    q = retrieval_indices[:,:k]
    count = 0
    matched_words = []
    img_dir = []
    words_len = []
    min_ext_len = 3
    # get all matched words
    for i in range(len(q)):
        matched = []
        for j in q[i]:
            matched.append(words[j])
            curr_len = len(words[j])
            curr_dir = word_var_dir[words[j]]
            if len(ground_truth[i]) < min_ext_len:
                curr_dir = 0
            words_len.append(curr_len + abs(curr_dir))
            img_dir.append(curr_dir)
        matched_words.append(matched)
    
    # calculate accuracies
    for i in range(len(ground_truth)):
        #print word_strings[i]
        if ground_truth[i].lower() in [mw.lower() for mw in matched_words[i]]:
            count = count+1
        else:
            for w in matched_words[i]:
                if ground_truth[i] in root_word_var and w in root_word_var[ground_truth[i]]:
                    count = count+1
                    break

    return (count, matched_words, qualified_ids, img_dir, words_len, outputs, ground_truth)


# In[30]:


# the old original report matches method
from scipy.spatial.distance import cdist, pdist, squareform
def report_matches(outputs, embedding, matching, words, ground_truth, k, length):
    # length sorting stuff
    qualified_ids = [x for x in range(len(ground_truth)) if len(ground_truth[x]) > length]
    outputs = np.array(outputs)
    ground_truth = np.array(ground_truth)
    outputs = list(outputs[qualified_ids])
    ground_truth = list(ground_truth[qualified_ids])
    # the real computation
    dist_mat = cdist(XA=outputs, XB=embedding, metric=matching)
    retrieval_indices = np.argsort(dist_mat, axis=1)
    q = retrieval_indices[:,:k]
    count = 0
    matched_words = []
    # get all matched words
    for i in range(len(q)):
        matched = []
        for j in q[i]:
            matched.append(words[j])
        matched_words.append(matched)
    
    for i in range(len(ground_truth)):
        if ground_truth[i].lower() in [mw.lower() for mw in matched_words[i]]:
            count = count+1

    return (count, matched_words, outputs, embedding, ground_truth, qualified_ids)


# In[31]:


# given the image name, this driver function computes the following
# 1. loads the words and images and transforms them based on image name
# 2. generates embeddings for images using the cnn model
# 3. gets the original and variation embeddings
# 4. generate report with word variations (prints accuracy)
# 5. generate report original (prints accuracy)
# 6. returns the image_dir_info that needs to be saved as numpy files
def image_ext_with_word_var(map_name, cnn, gis_data, text_phoc_info, global_stats):
    images, words = load_and_transform(map_name)
    img_phoc_embs = gen_img_phoc_embs(cnn, images)
    embedding, embedding_var, word_var_strings, word_var_dir, root_word_var, conf_set = text_phoc_info
    original_report = report_matches(img_phoc_embs, embedding, 'cosine', gis_data, words, 1, word_filter_len)
    global_stats['correct_original'] += original_report[0]
    print 'Original Accuracy ', str(original_report[0]/float(len(original_report[4])))
    word_var_report = report_matches_with_variations(img_phoc_embs, embedding_var,'cosine', words, word_var_strings, word_var_dir, root_word_var, 1, word_filter_len)
    global_stats['correct_word_var'] += word_var_report[0]
    print 'Accuracy With Word Variations ', str(word_var_report[0]/float(len(word_var_report[4])))
    global_stats['total'] += len(word_var_report[4])
    img_dir_info = np.array([word_var_report[2], word_var_report[3], word_var_report[4]])
    return img_dir_info, word_var_report[6], conf_set
    


# In[32]:


gis_data = load_and_clean_gis_data()
text_phoc_info = gen_text_phoc_embs(gis_data)

import sys
map_name = sys.argv[1]

global_stats = {'correct_original':0, 'correct_word_var':0, 'total':0}

print map_name
img_dir_info, words, conf_words = image_ext_with_word_var(map_name, cnn, gis_data, text_phoc_info, global_stats)
np.save('../../../images_to_extend/ray_output_gis/org_img_ext_2var/image_dir_'+map_name+'.npy', img_dir_info)
np.save('../../../images_to_extend/ray_output_gis/org_img_ext_2var/image_labels_'+map_name+'.npy', words)

print 'Accuracy Original', global_stats['correct_original']/float(global_stats['total'])
print 'Accuracy With Word Variations', global_stats['correct_word_var']/float(global_stats['total'])
print global_stats