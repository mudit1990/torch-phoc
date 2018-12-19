'''
The notebook takes all the images & words in a Map, creates word variations, 
runs phoc and gets the closest match to every clip in the map. 
Based on the matched word variation we get the direction the clip should 
be extended in and store it in a separate file. Below cells also contain
some analysis on how phoc performs, the top 10 neighbors etc
'''

import argparse
import logging

import numpy as np
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
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


model_path = 'models/PHOCNet_Nov13.pt' # path of the phoc model to use
# the set of maps the code will be run on
A = ['D0042-1070001', 'D0042-1070002', 'D0042-1070006', 'D0042-1070007', 'D0117-5755018', 'D0117-5755035', 'D0117-5755036']

word_filter_len = 1 # only words above this length are considered valid
max_var_len = 1


if not torch.cuda.is_available():
    logger.warning('Could not find CUDA environment, using CPU mode')
    gpu_id = None
else:
    gpu_id = [0]
#torch.cuda.get_device_name(gpu_id[0])
pass


model_ = torch.load(model_path)
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
    # images = np.load('../../../ProcessedData/original_images_nopad_'+map_name+'.tiff.npy')
    # words = np.load('../../../ProcessedData/original_words_nopad_'+map_name+'.tiff.npy')
    images = np.transpose(images, (0,3,1,2))
    words = clean_words(words)
    print 'Images Shape ', images.shape
    print 'Words Shape ', words.shape
    return images, words


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
# word_var is a dictionary that contains all variations as key and 0,1,-1 as value
# 0 denotes the root word, -1 denotes var = root_word[:-1], +1 denotes var = root_word[1:]
# root_word_var is a dict that stores original_word => all_variations
# enable_conf: boolean flag that controls if the confusion logic should be used.
# when enabled if a word is a root word as well as a word variation (happens if root words ar rand and grand)
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


# the new report matches method that handles variations
from scipy.spatial.distance import cdist, pdist, squareform
def report_matches_with_variations(outputs, embedding_var, matching, word_strings, 
                                   word_var_strings, word_var_dir, root_word_var, k, length):
    # length sorting stuff
    qualified_ids = [x for x in range(len(word_strings)) if len(word_strings[x]) > length]
    outputs = np.array(outputs)
    word_strings = np.array(word_strings)
    outputs = list(outputs[qualified_ids])
    word_strings = list(word_strings[qualified_ids])
    
    # same stuff for variations
#     qualified_ids_vars = [x for x in range(len(word_var_strings)) if len(word_var_strings[x]) > (length)]
#     embedding_var = np.array(embedding_var)
#     word_var_strings = np.array(word_var_strings)
#     embedding_var = list(embedding_var[qualified_ids_vars])
#     word_var_strings = list(word_var_strings[qualified_ids_vars])
    
    # the real computation
    dist_mat = cdist(XA=outputs, XB=embedding_var, metric=matching)
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
            matched.append(word_var_strings[j])
            curr_len = len(word_var_strings[j])
            curr_dir = word_var_dir[word_var_strings[j]]
            # don't extend an image if its ground truth length is less than min_ext_len
            if len(word_strings[i]) < min_ext_len:
                curr_dir = 0
            words_len.append(curr_len + abs(curr_dir))
            img_dir.append(curr_dir)
        matched_words.append(matched)
    
    # calculate accuracies
    for i in range(len(word_strings)):
        #print word_strings[i]
        if word_strings[i].lower() in [mw.lower() for mw in matched_words[i]]:
            count = count+1
        else:
            for w in matched_words[i]:
                if w in root_word_var[word_strings[i]]:
                    count = count+1
                    break

    return (count, matched_words, qualified_ids, img_dir, words_len, outputs, word_strings, dist_mat)


# the old original report matches method
from scipy.spatial.distance import cdist, pdist, squareform
def report_matches(outputs, embedding, matching, word_strings, k, length):
    # length sorting stuff
    qualified_ids = [x for x in range(len(word_strings)) if len(word_strings[x]) > length]
    outputs = np.array(outputs)
    embedding = np.array(embedding)
    word_strings = np.array(word_strings)
    outputs = list(outputs[qualified_ids])
    embedding = list(embedding[qualified_ids])
    word_strings = list(word_strings[qualified_ids])
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
            matched.append(word_strings[j])
        matched_words.append(matched)
    
    for i in range(len(word_strings)):
        if word_strings[i].lower() in [mw.lower() for mw in matched_words[i]]:
            count = count+1

    return (count, matched_words, outputs, embedding, word_strings, qualified_ids, dist_mat)


# given the image name, this driver function computes the following
# 1. loads the words and images and transforms them based on image name
# 2. generates embeddings for images using the cnn model
# 3. gets the original and variation embeddings
# 4. generate report with word variations (prints accuracy)
# 5. generate report original (prints accuracy)
# 6. returns the image_dir_info that needs to be saved as numpy files
def image_ext_with_word_var(map_name, cnn, global_stats):
    images, words = load_and_transform(map_name)
    img_phoc_embs = gen_img_phoc_embs(cnn, images)
    print 'image shape before',images.shape
    embedding, embedding_var, word_var_strings, word_var_dir, root_word_var, conf_set = gen_text_phoc_embs(words)
    print set([s.lower() for s in conf_set])
    original_report = report_matches(img_phoc_embs, embedding, 'cosine', words, 1, word_filter_len)
    print 'image_shape_after_var', np.array(original_report[2]).shape
    global_stats['correct_original'] += original_report[0]
    print 'Original Accuracy ', str(original_report[0]/float(len(original_report[4])))
    word_var_report = report_matches_with_variations(img_phoc_embs, embedding_var,'cosine', words,                                                      word_var_strings, word_var_dir, root_word_var, 1, word_filter_len)
    print 'image_shape_after_var', np.array(word_var_report[5]).shape
    global_stats['correct_word_var'] += word_var_report[0]
    print 'Accuracy With Word Variations ', str(word_var_report[0]/float(len(word_var_report[4])))
    global_stats['total'] += len(word_var_report[4])
    img_dir_info = np.array([word_var_report[2], word_var_report[3], word_var_report[4]])
    return img_dir_info, word_var_report[6], conf_set, original_report, word_var_report


global_stats = {'correct_original':0, 'correct_word_var':0, 'total':0}
for i in tqdm(range(len(A)), ascii=True, desc = 'Main Iteration'):
    print A[i]
    img_dir_info, words, conf_words, original_report, word_var_report = image_ext_with_word_var(A[i], cnn, global_stats)
    # np.save('/mnt/nfs/work1/696ds-s18/mbhargava/images_to_extend/final_runs/ray_output_normal/image_dir_'+A[i]+'.npy', img_dir_info)
    # np.save('/mnt/nfs/work1/696ds-s18/mbhargava/images_to_extend/final_runs/ray_output_normal/image_labels_'+A[i]+'.npy', words)
print 'Average Accuracy Original', global_stats['correct_original']/float(global_stats['total'])
print 'Average Accuracy With Word Variations', global_stats['correct_word_var']/float(global_stats['total'])
