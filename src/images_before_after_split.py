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
    images_before = np.load('../../../detection_outputs_ready_for_test/ray_regions/org_clips/'+map_name+'.npy')
    words_before = np.load('../../../detection_outputs_ready_for_test/ray_labels/org_clips/'+map_name+'.npy')
    words_before = clean_words(words_before)
    images_before, words_before = clean_word_images(images_before, words_before)
    images_before = np.transpose(images_before, (0,3,1,2))
    
    images_after = np.load('../../../detection_outputs_ready_for_test/ray_regions_gis/'+map_name+'.npy')
    words_after = np.load('../../../detection_outputs_ready_for_test/ray_labels_gis/'+map_name+'.npy')
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

A = ['D0042-1070006','D0117-5755018','D0117-5755035','D0117-5755035']

for i in range(len(A)):
    print 'running for image', A[i]
    images_before, images_after, words = load_and_transform(A[i])
    mid = int(len(images_before)/2)
    np.save('image_splits/before/'+A[i]+'_1.npy', images_before[:mid])
    np.save('image_splits/after/'+A[i]+'_1.npy', images_after[:mid])
    np.save('image_splits/words/'+A[i]+'_1.npy', words[:mid])

    np.save('image_splits/before/'+A[i]+'_2.npy', images_before[mid:])
    np.save('image_splits/after/'+A[i]+'_2.npy', images_after[mid:])
    np.save('image_splits/words/'+A[i]+'_2.npy', words[mid:])
