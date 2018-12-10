from __future__ import division
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

# load before, after images and words, transforms and cleans them
# the function also assumes that ground truth words are the same before and after
# returns before_images, after_images, words
def load_and_transform(map_name):
    images_before = np.load('/mnt/nfs/work1/696ds-s18/mbhargava/detection_outputs_ready_for_test/ray_regions/org_clips/'+map_name+'.npy')
    words_before = np.load('/mnt/nfs/work1/696ds-s18/mbhargava/detection_outputs_ready_for_test/ray_labels/org_clips/'+map_name+'.npy')
    words_before = clean_words(words_before)
    images_before, words_before = clean_word_images(images_before, words_before)
    images_before = np.transpose(images_before, (0,3,1,2))
    
    images_after = np.load('/mnt/nfs/work1/696ds-s18/mbhargava/detection_outputs_ready_for_test/ray_regions_gis/extend_by2/'+map_name+'.npy')
    words_after = np.load('/mnt/nfs/work1/696ds-s18/mbhargava/detection_outputs_ready_for_test/ray_labels_gis/extend_by2/'+map_name+'.npy')
    words_after = clean_words(words_after)
    images_after, words_after = clean_word_images(images_after, words_after)
    images_after = np.transpose(images_after, (0,3,1,2))
    
    print 'Images Before Shape ', images_before.shape
    print 'Words Before Shape ', words_before.shape
    print 'Images After Shape ', images_after.shape
    print 'Words After Shape ', words_after.shape
    return images_before, images_after, words_after

def clean_word_images(images, words):
    selected_idx = [x for x in range(len(words)) if len(words[x]) > word_filter_len]
    images = images[selected_idx]
    words = words[selected_idx]
    return images, words

def load_and_clean_gis_data():
    with open('../../../GIS_data/GIS_combined.txt') as f:
        gis_data = np.array(f.read().splitlines())
    gis_data = clean_words(gis_data)
    print 'GIS Data', gis_data.shape
    return gis_data


# convert image tnto embedding using the cnn model
def get_image_embeddings(cnn, images):
    outputs = []
    for i in tqdm(range(len(images))):
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


# compute the PHOC representation of the word itself
from cnn_ws.string_embeddings.phoc import build_phoc_descriptor
def get_word_phoc_representations(word_strings):
    unigrams = [chr(i) for i in range(ord('&'), ord('&')+1) + range(ord('A'), ord('Z')+1) + range(ord('a'), ord('z') + 1) + range(ord('0'), ord('9') + 1)]
    bigram_levels = None
    bigrams = None
    phoc_unigram_levels=(1, 2, 4, 8)
    word_var_dir, root_word_var, conf_words = create_word_variations(word_strings, enable_conf=True)
    
    word_var_strings = word_var_dir.keys()
    embedding_var = build_phoc_descriptor(words=word_var_strings,
                                  phoc_unigrams=unigrams,
                                  bigram_levels=bigram_levels,
                                  phoc_bigrams=bigrams,
                                  unigram_levels=phoc_unigram_levels)
    
    print('embedding variations:', embedding_var.shape)
    return (embedding_var, word_var_strings, word_var_dir, root_word_var, conf_words)



from scipy.spatial.distance import cdist, pdist, squareform

# gets the actual distances of all the ground truth word variations
def get_all_dist_gt(dist_mat, emb_info, ground_truth):
    # expand emb_info tuple
    embedding_var, word_var_strings, word_var_dir, root_word_var,_ = emb_info
    all_dist = []
    for i in range(len(ground_truth)):
        w_dist = []
        if ground_truth[i] in root_word_var:
            w_vars = root_word_var[ground_truth[i]]
            for j in range(len(word_var_strings)):
                if word_var_strings[j] in w_vars:
                    w_dist.append((word_var_strings[j], dist_mat[i][j]))
        all_dist.append(w_dist)
    return all_dist

# the new report matches method that handles variations
def report_matches_with_variations(dist_mat, ground_truth, emb_info, k):
    # expand emb_info tuple
    embedding_var, word_var_strings, word_var_dir, root_word_var,_ = emb_info
    gt_words_dist = []
    # gt_words_dist = get_all_dist_gt(dist_mat, emb_info, ground_truth)
    retrieval_indices = np.argsort(dist_mat, axis=1)
    q = retrieval_indices[:,:k]
    count = 0
    matched_words = []
    img_dir = []
    words_len = []
    actual_dist = []
    min_ext_len = 3
    # get all matched words
    for i in range(len(q)):
        matched = []
        for j in q[i]:
            actual_dist.append(dist_mat[i][j])
            matched.append(word_var_strings[j])
            curr_len = len(word_var_strings[j])
            curr_dir = word_var_dir[word_var_strings[j]]
            if len(ground_truth[i]) < min_ext_len:
                curr_dir = 0
            words_len.append(curr_len + abs(curr_dir))
            img_dir.append(curr_dir)
        matched_words.append(matched)
    
    # calculate accuracies
    is_correct = []
    for i in range(len(ground_truth)):
        #print word_strings[i]
        is_correct.append(0)
        if ground_truth[i].lower() in [mw.lower() for mw in matched_words[i]]:
            is_correct[i] = 1
            count = count+1
        else:
            for w in matched_words[i]:
                if ground_truth[i] in root_word_var and w in root_word_var[ground_truth[i]]:
                    is_correct[i] = 2
                    count = count+1
                    break
    return (count, matched_words, img_dir, words_len, actual_dist, is_correct, gt_words_dist)

# For some images, the original predicted word os both a root word and a word_variation of another word 
# (common word problem). Due to this one cannot be sure, if these images should be extended or not.
# These images are handled by comparing distances before and after image extension and picking the minimum one
# the feature can be turned of by setting enable_conf = False
def update_dist_matrix(dist_mat_before, dist_mat_after, conf_idx):
    print('conf_idx', conf_idx)
    for i in conf_idx:
        dist = np.minimum(dist_mat_before[i], dist_mat_after[i])
        dist_mat_after[i] = dist



def generate_confusion_matrix(match_report_before, match_report_after, words):
    status = [0,1,2]
    conf_matrix = [['before/after','incorrect (0)','correct (1)','almost (2)'],
                   ['incorrect (0)',0,0,0],
                   ['correct (1)',0,0,0],
                   ['almost (2)',0,0,0]]
    for i in status:
        for j in status:
            count = 0
            for k in range(len(words)):
                if match_report_before[5][k] == i and match_report_after[5][k] == j:
                    count += 1
            conf_matrix[1+i][1+j] = count
    return conf_matrix


def save_before_after_preds(map_name, before_report, after_report, ground_truth):
    before_preds = [w[0] for w in before_report[1]]
    after_preds = [w[0] for w in after_report[1]]
    before_after_pred = np.array([before_preds, after_preds, ground_truth]).T
    np.save('../../../before_after_ext_pred/ray_gis_detections/extend_by2/'+map_name+'.npy', before_after_pred)

def compare_images_before_after_ext(map_name, cnn, global_stats, word_emb_info):
    images_before, images_after, words = load_and_transform(map_name)
    image_embs_before = get_image_embeddings(cnn, images_before)
    image_embs_after = get_image_embeddings(cnn, images_after)
    # get the distances between images and words
    dist_matrix_before = cdist(XA=image_embs_before, XB=word_emb_info[0], metric='cosine')
    print 'before distance calculation done'
    dist_matrix_after = cdist(XA=image_embs_after, XB=word_emb_info[0], metric='cosine')
    print 'after distance calculation done'
    # build the original report
    match_report_before = report_matches_with_variations(dist_matrix_before, words, word_emb_info, 1)
    # get the low confidence image index
    conf_idx = [i for i in range(len(match_report_before[1])) if match_report_before[1][i][0] in word_emb_info[4]]
    # update the dist_after matrix based for low confidence images
    update_dist_matrix(dist_matrix_before, dist_matrix_after, conf_idx)
    # build the report after extension
    match_report_after = report_matches_with_variations(dist_matrix_after, words, word_emb_info, 1)
    save_before_after_preds(map_name, match_report_before, match_report_after, words)
    global_stats['correct_before'] += match_report_before[0]
    global_stats['correct_after'] += match_report_after[0]
    global_stats['total'] += len(words)
    acc_before = match_report_before[0]/len(words)
    acc_after = match_report_after[0]/len(words)
    conf_matrix = generate_confusion_matrix(match_report_before, match_report_after, words)
    return (acc_before, acc_after, conf_matrix, match_report_before, match_report_after)


import sys
map_name = sys.argv[1]
print map_name


gis_data = load_and_clean_gis_data()
text_phoc_info = get_word_phoc_representations(gis_data)


global_stats = {'correct_before':0, 'correct_after':0, 'total':0}
stats = compare_images_before_after_ext(map_name, cnn, global_stats, text_phoc_info)

# import pickle
# pickle.dump(stats, open('../../../before_after_outputs/'+map_name,'wb'))

print "the accuracy before extension: " + str(stats[0])
print "the accuracy after extension: "+str(stats[1])
print global_stats