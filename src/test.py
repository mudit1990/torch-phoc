import argparse
import logging

import numpy as np
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import copy
from datasets.maps_alt import MAPSDataset
#from maps_alt import MAPSDataset

#from cnn_ws.transformations.homography_augmentation import HomographyAugmentation
from cnn_ws.losses.cosine_loss import CosineLoss

from cnn_ws.models.myphocnet import PHOCNet
from cnn_ws.evaluation.retrieval import map_from_feature_matrix, map_from_query_test_feature_matrices
from torch.utils.data.dataloader import _DataLoaderIter as DataLoaderIter
from torch.utils.data.sampler import WeightedRandomSampler

from cnn_ws.utils.save_load import my_torch_save, my_torch_load

import matplotlib.pyplot as plt
from cnn_ws.string_embeddings.phoc import build_phoc_descriptor

from scipy.spatial.distance import cdist, pdist, squareform
#from dist_func import damerau_levenshtein_distance as dld

def report_matches(outputs, embedding, matching, word_strings, original_words, k, length, is_lower):
    # length sorting stuff
    qualified_ids = [x for x in range(len(word_strings)) if len(word_strings[x]) > length]
    qualified_ids_original = [x for x in range(len(original_words)) if len(original_words[x]) > length]
    outputs = np.array(outputs)
    embedding = np.array(embedding)
    word_strings = np.array(word_strings)
    original_words = np.array(original_words)
    outputs = list(outputs[qualified_ids_original])
    embedding = list(embedding[qualified_ids])
    word_strings = list(word_strings[qualified_ids])
    original_words = list(original_words[qualified_ids_original])
    # the real computation
    dist_mat = cdist(XA=outputs, XB=embedding, metric=matching)
    retrieval_indices = np.argsort(dist_mat, axis=1)
    q = retrieval_indices[:,:k]
    count = 0
    matched_words = []
    # get all matched words
    #print len(outputs), len(embedding)
    for i in range(len(q)):
        matched = []
        #print q[i]
        for j in q[i]:
            matched.append(word_strings[j])
        matched_words.append(matched)
    
    #print len(word_strings), len(matched_words)
    close_counts = 0
    for i in range(len(original_words)):
        #print original_words[i], matched_words[i]
        if is_lower:
            if original_words[i] in matched_words[i]:
                #print "yes"
                count = count+1
        else:
            if original_words[i].lower() in [x.lower() for x in matched_words[i]]:
                count = count+1
                
    #print count
    return count, close_counts, matched_words, outputs, embedding, word_strings, qualified_ids_original

def learning_rate_step_parser(lrs_string):
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for elem in lrs_string.split(',')]

# parameters ######################################################################################
embedding_type = 'phoc'
bigram_levels = None
bigrams = None
phoc_unigram_levels = (1,2,4,8)
fixed_image_size = None
min_image_width_height = 26
is_lower = 0

# get the test set ################################################################################
# f = open('../splits/val_files.txt', 'rb')
# all_files = f.readlines()
# all_files = [x.strip('\n') for x in all_files]
# f.close()

# test_set = MAPSDataset(map_root_dir1='../../../ProcessedData/',
#                 map_root_dir2='../../../ProcessedData/',
#                 all_files=all_files,
#                 embedding=embedding_type,
#                 phoc_unigram_levels=phoc_unigram_levels,
#                 fixed_image_size=fixed_image_size,
#                 min_image_width_height=min_image_width_height, is_lower=is_lower)

if not torch.cuda.is_available():
    logger.warning('Could not find CUDA environment, using CPU mode')
    gpu_id = None
else:
    gpu_id = [0]

# test_set.mainLoader(partition='test', transforms=None)
# test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

# get the model ####################################################################################
#cnn = PHOCNet(n_out=list(test_set[0][1].size())[0],
#    input_channels=3,
#    gpp_type='gpp',
#    pooling_levels=([1], [5]))

#cnn.init_weights()
#right now a1 is the best model
if is_lower == 1:
    model_ = torch.load('models/PHOCNet_final.pt')
else:
    model_ = torch.load('models/PHOCNet_a.pt')

cnn = model_.module
cnn.training = False
if gpu_id is not None:
        if len(gpu_id) > 1:
            cnn = nn.DataParallel(cnn, device_ids=gpu_id)
            cnn.cuda()
        else:
            cnn.cuda(gpu_id[0])

# find the file names
f = open('../splits/val_files.txt', 'rb')
A = f.readlines()
f.close()
A = [x.rstrip('\n') for x in A]
# A.remove('D0042-1070013')

avg_accuracy = 0
global_total = 0
global_correct = 0
# with open('../../../GIS_data/GIS_combined.txt') as f:
#     words = np.array(f.read().splitlines())
# print words

for i in tqdm(range(0, len(A))):
	# load test images and words
	images = np.load('../../../detection_outputs_ready_for_test/detected_regions/'+A[i]+'.npy')
	words = np.load('../../../detection_outputs_ready_for_test/detected_labels/'+A[i]+'.npy')
	
	original_words = np.copy(words)
	
	if is_lower == 1:
    		pass
	else:    
    		words = list(words)
    		new_words = []
    		for i in words:
        		new_words.append(i.upper())
        		new_words.append(i.lower())
        		new_words.append(i.capitalize())
    	words = np.array(new_words)
	print len(words)
	# convert dimensions
	images = np.transpose(images, (0,3,1,2))
	images.shape
	
	# check if this works
	outputs = []
	for i in range(len(images)):
    		word_img = images[i]
    		word_img = 1 - word_img.astype(np.float32) / 255.0
    		word_img = word_img.reshape((1,) + word_img.shape)
    		word_img = torch.from_numpy(word_img).float()
    		word_img = word_img.cuda(gpu_id[0])
    		word_img = torch.autograd.Variable(word_img)
    		output = torch.sigmoid(cnn(word_img))
    		output = output.data.cpu().numpy().flatten()
    		outputs.append(output)
	
	# compute the PHOC representation of the word itself
	word_strings = words
	if is_lower == 0:
    		unigrams = [chr(i) for i in range(ord('&'), ord('&')+1) + range(ord('A'), ord('Z')+1) + \
                    range(ord('a'), ord('z') + 1) + range(ord('0'), ord('9') + 1)]
	else:
    		unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1) + range(ord('0'), ord('9') + 1)]


	if is_lower == 1:
    		for i in range(len(word_strings)):
        		word_strings[i] = word_strings[i].lower()
	else:
    		pass

	embedding = build_phoc_descriptor(words=word_strings,
                                  phoc_unigrams=unigrams,
                                  bigram_levels=bigram_levels,
                                  phoc_bigrams=bigrams,
                                  unigram_levels=phoc_unigram_levels)

	print embedding.shape

	count, close_count, matched_words, new_outputs, new_embedding, new_word_strings, \
        qualified_ids = report_matches(outputs, embedding, 'cosine', word_strings, \
                                       original_words, k=1, length=3, is_lower=is_lower)

	print "the accuracy is: "+str(count/float(len(qualified_ids)))
	global_total += len(qualified_ids)
	global_correct += count
	#print "the close_count accuracy is: "+str(close_count/float(len(original_words)))
	
	avg_accuracy += count/float(len(qualified_ids))

print 'correct average acc: ' + str(global_correct/float(global_total))
print "average accuracy is: "+str(avg_accuracy/len(A))
