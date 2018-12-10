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


# In[23]:


if not torch.cuda.is_available():
    logger.warning('Could not find CUDA environment, using CPU mode')
    gpu_id = None
else:
    gpu_id = [0]
#torch.cuda.get_device_name(gpu_id[0])
pass


# In[24]:


model_ = torch.load('models/PHOCNet_final.pt')
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
    images = np.load('../../../detection_outputs_ready_for_test/ray_regions/'+map_name+'.npy')
    words = np.load('../../../detection_outputs_ready_for_test/ray_labels/'+map_name+'.npy')
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


# In[27]:


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


def gen_text_phoc_embs(words):
    word_strings = [w.lower() for w in words]
    unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1) + range(ord('0'), ord('9') + 1)]
    bigram_levels = None
    bigrams = None
    phoc_unigram_levels=(1, 2, 4, 8)
    
    embedding = build_phoc_descriptor(words=word_strings,
                                  phoc_unigrams=unigrams,
                                  bigram_levels=bigram_levels,
                                  phoc_bigrams=bigrams,
                                  unigram_levels=phoc_unigram_levels)
    
    return embedding


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
def image_ext_with_word_var(map_name, cnn, gis_data, embedding, global_stats):
    images, words = load_and_transform(map_name)
    img_phoc_embs = gen_img_phoc_embs(cnn, images)
    original_report = report_matches(img_phoc_embs, embedding, 'cosine', gis_data, words, 1, word_filter_len)
    global_stats['correct_original'] += original_report[0]
    print 'Original Accuracy ', str(original_report[0]/float(len(original_report[4])))
    global_stats['total'] += len(original_report[4])
    


# In[32]:

import sys
map_name = sys.argv[1]

gis_data = load_and_clean_gis_data()
embedding = gen_text_phoc_embs(gis_data)


global_stats = {'correct_original':0, 'correct_word_var':0, 'total':0}
image_ext_with_word_var(map_name, cnn, gis_data, embedding, global_stats)
    # np.save('../../../images_to_extend/ray_output_gis/image_dir_'+A[i]+'.npy', img_dir_info)
    # np.save('../../../images_to_extend/ray_output_gis/image_labels_'+A[i]+'.npy', words)
print 'Accuracy Original', global_stats['correct_original']/float(global_stats['total'])
print 'Accuracy With Word Variations', global_stats['correct_word_var']/float(global_stats['total'])
print global_stats