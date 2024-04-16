import argparse
import os
import pickle
from transformers import BertTokenizer
from os import listdir
from os.path import isfile, join
from io import open
import torch
import sys
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.autograd import Variable
import time
import math
from math import sin, cos, sqrt, atan2, radians
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from ast import literal_eval
import random
from difflib import SequenceMatcher
import config

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
    
    
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def compute_dist(lat1, lon1, lat2, lon2):

    R = 6373.0
    
    try:
        float(lat1)
    except ValueError:
        return -1
        
    try:
        float(lon1)
    except ValueError:
        return -1
        
    try:
        float(lat2)
    except ValueError:
        return -1
        
    try:
        float(lon2)
    except ValueError:
        return -1
        
        
    
    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
        
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))
                
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    dist = round(R * c * 1000)
    dist = 2*(dist)/config.MAX_DIST - 1
    
    return dist
    
    
def get_lat_long(entity):

  words = entity.lower().split()
  for i,word in enumerate(words):
    if words[i-2] == 'latitude' and words[i-1] == 'val':
      latitude = float(word)
      longitude = float(words[i+4])
      idx = i

    if words[i-2] == 'postalcode' and words[i-1] == 'val':
      try:
        words[i] = str(int(float(words[i])))
      except ValueError:
        pass

  del words[idx-3:idx+5]


  return ' '.join(words), latitude, longitude

    
def prepare_dataset(path, n_path, max_seq_len=128):
  np.random.seed(config.seed)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  data_x = []
  coord_x = []
  data_y = []
  neigh_x = []

  with open(path, 'r') as f:
    
    for line in f:
      
      arr = line.split('\t')

      e1, lat1, long1 = get_lat_long(arr[0])
      e2, lat2, long2 = get_lat_long(arr[1])
  
      if len(arr) > 2:

        x = tokenizer.tokenize('[CLS] ' + e1 +' [SEP] ' + e2 + ' [SEP]')
        y = arr[2]

        if len(x) < max_seq_len:
          x = x + ['[PAD]']*(max_seq_len - len(x))
        else:
          x = x[:max_seq_len]

        x = custom_tokenize(tokenizer.convert_tokens_to_ids(x),tokenizer.vocab_size,config.add_noise)
        data_x.append(x)
        coord_x.append(compute_dist(lat1, long1, lat2, long2))
        data_y.append(int(y.strip()))
        


  with open(n_path, encoding='utf-8') as f:
    neigh = json.load(f)
  add_neighbor_noise = config.add_neighbor_noise
  onlyname = config.onlyname
  for b in range(len(neigh)):
    neigh[b]['name1'] = custom_tokenize(tokenizer(neigh[b]['name1'])['input_ids'],tokenizer.vocab_size,config.add_noise)
    neigh[b]['name2'] = custom_tokenize(tokenizer(neigh[b]['name2'])['input_ids'],tokenizer.vocab_size,config.add_noise)
    neigh[b]['attr1'] = custom_tokenize(tokenizer(neigh[b]['attr1'])['input_ids'],tokenizer.vocab_size,config.add_noise,onlyname)
    neigh[b]['attr2'] = custom_tokenize(tokenizer(neigh[b]['attr2'])['input_ids'],tokenizer.vocab_size,config.add_noise,onlyname)
    neigh[b]['neigh1_attr'] = [custom_tokenize(tokenizer(n)['input_ids'],tokenizer.vocab_size,add_neighbor_noise,onlyname) for n in neigh[b]['neigh1_attr']]
    neigh[b]['neigh2_attr'] = [custom_tokenize(tokenizer(n)['input_ids'],tokenizer.vocab_size,add_neighbor_noise,onlyname) for n in neigh[b]['neigh2_attr']]
    neigh[b]['neigh1'] = [custom_tokenize(tokenizer(n)['input_ids'],tokenizer.vocab_size,add_neighbor_noise) for n in neigh[b]['neigh1']]
    neigh[b]['neigh2'] = [custom_tokenize(tokenizer(n)['input_ids'],tokenizer.vocab_size,add_neighbor_noise) for n in neigh[b]['neigh2']]

  return data_x, coord_x, neigh, data_y


def find_second_occurrence(lst, value):
    count = 0
    for index, item in enumerate(lst):
        if item == value:
            count += 1
            if count == 2:
                return index
    return len(lst)
        
def custom_tokenize(tokens,vocab_size,add_noise,onlyname=False):
  skip_token_list = [101,102,103,0,8902,11748]
  name_ind = find_second_occurrence(tokens,8902)
  if add_noise:
      for i in range(0,len(tokens)):
          if i > name_ind and onlyname:
            break
          if tokens[i] in skip_token_list:
              continue
          else:
              prob = config.global_noise_prob
          if np.random.rand()<prob:
              if np.random.rand() < 0.5:
                  tokens[i]=np.random.randint(vocab_size)
              else:
                  tokens[i]=103 #mask_id
  return tokens

  

