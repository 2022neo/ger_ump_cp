import argparse
import os
from os import listdir
from os.path import isfile, join
from io import open
import torch
import sys
from torch import nn, optim
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer
import numpy as np
from collections import Counter
from torch.autograd import Variable
import random
import time
import math
from math import sin, cos, sqrt, atan2, radians
import json
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from ast import literal_eval
from difflib import SequenceMatcher
from pathlib import Path


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
    
    
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def get_data_dict(words):
    data_dict={'name':[],'address':[],'postalcode':[],'latitude':[],'longitude':[],'categories':[]}
    for i in range(len(words)):
        if words[i-1] == 'val' and words[i-2] in data_dict:
            j=i
            data = data_dict[words[i-2]]
            while j < len(words) and not (words[j]=='col'):
                data.append(words[j])
                j+=1
    data_dict1={}    
    for key in data_dict:
        new_key = 'postalCode' if key=='postalcode' else key
        data_dict1[new_key] = ' '.join(data_dict[key]) if len(data_dict[key])>0 else ' '
        data_dict1[new_key] = float(data_dict1[new_key]) if new_key=='latitude' or new_key=='longitude' else data_dict1[new_key]
    return data_dict1

def dict_to_sentence(dictionary):
    sentence = ''
    for k,v in dictionary.items():
        if k=='longitude' or k == 'latitude':
            continue
        if  (v ==' '):
            pass
        if not isinstance(v,str):
            print(k,v,type(v),bool(v))
        sentence += 'COL '+ k + ' VAL ' + v + ' '
    return sentence.strip()

def compute_dist(lat1, lon1, lat2, lon2):

    R = 6373.0
    
    try:
        float(lat1)
    except ValueError:
        return ' '
        
    try:
        float(lon1)
    except ValueError:
        return ' '
        
    try:
        float(lat2)
    except ValueError:
        return ' '
        
    try:
        float(lon2)
    except ValueError:
        return ' '
        
        
    
    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
        
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))
                
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return str(round(R * c * 1000))

# parser = argparse.ArgumentParser(description='GeoER')
# parser.add_argument('-s', '--source', type=str, default='osm_yelp', help='Data source (oms_yelp, osm_fsq')
# parser.add_argument('-c', '--city', type=str, default='pit', help='City dataset (sin, edi, tor, pit)')
# args = parser.parse_args()

# city = f'{args.city}'
# split = f'/{args.source}/'

NEIGHBORHOOD_RADIUS = 1000
HIDDEN_SIZE = 768


def main(param):
    city,split = param
    csv_root = '/home/jiyuwen/project/ger_refiner/pretrain_data'
    dataset_table_path = csv_root + split + city+ '.csv'
    dataset = pd.read_csv(dataset_table_path,index_col=0).fillna(' ')


    train_path = 'train_valid_test'+split+city+'/train.txt'
    valid_path = 'train_valid_test'+split+city+'/valid.txt'
    test_path = 'train_valid_test'+split+city+'/test.txt'

    train_path_out = 'ablation_neighborhood_train_valid_test/'+split+city+'/n_train.json'
    valid_path_out = 'ablation_neighborhood_train_valid_test/'+split+city+'/n_valid.json'
    test_path_out = 'ablation_neighborhood_train_valid_test/'+split+city+'/n_test.json'

    for path in [train_path, valid_path, test_path]:

        entries = []

        if path == train_path:
            out_path = train_path_out
            print('Preparing train neighborhood data...')
        elif path == valid_path:
            out_path = valid_path_out
            print('Preparing valid neighborhood data...')
        else:
            out_path = test_path_out
            print('Preparing test neighborhood data...')
        if not Path(out_path).exists():
            Path(out_path).parent.mkdir(exist_ok=True,parents=True)
        else:
            print('test neighborhood data existed !')
            continue
        
        count = 0
        with open(path, 'r') as f:
            for line in f:
                count+=1

        c = 0
        with open(path, 'r') as f:
        
            for line in f:
            
                e1 = line.split('\t')[0].lower()
                e2 = line.split('\t')[1].lower()
                
                entry = {}
                
                name1 = []
                name2 = []
                
                words = e1.split()
                data_dict1 = get_data_dict(words)
                name1 = data_dict1['name']
                attr1 = dict_to_sentence(data_dict1)
                lat1,long1 = str(data_dict1['latitude']),str(data_dict1['longitude'])

                
                
                words = e2.split()
                data_dict2 = get_data_dict(words)
                name2 = data_dict2['name']
                attr2 = dict_to_sentence(data_dict2)
                lat2,long2 = str(data_dict2['latitude']),str(data_dict2['longitude'])
            
                
                neighborhood1 = []
                neighborhood2 = []
                
                distances1 = []
                distances2 = []


                entry['name1'] = name1
                entry['name2'] = name2
                entry['attr1'] = attr1
                entry['attr2'] = attr2
                entry['pos1'] = [long1,lat1]
                entry['pos2'] = [long2,lat2]
                
                entry['neigh1'] = []
                entry['neigh2'] = []
                entry['neigh1_attr'] = []
                entry['neigh2_attr'] = []
                entry['dist1'] = []
                entry['dist2'] = []
                entry['neigh1_pos'] = []
                entry['neigh2_pos'] = []

                
                
                for i in range(dataset.shape[0]):
                    row = dataset.iloc[i]
                    
                    dist = compute_dist(lat1, long1, str(row['latitude']), str(row['longitude']))

                    try:
                        dist = int(dist)
                    except ValueError:
                        continue
                    
                    neighbor_attr = dict_to_sentence(row)
                    if  dist < NEIGHBORHOOD_RADIUS and len(entry['neigh1'])<50:
                        if (name1 == row['name'].lower() and str(row['latitude']) == lat1 and str(row['longitude']) == long1) or (name2 == row['name'].lower() and str(row['latitude']) == lat2 and str(row['longitude']) == long2):
                            continue
                        entry['neigh1'].append(row['name'].lower())
                        entry['neigh1_attr'].append(neighbor_attr.lower())
                        entry['dist1'].append(dist)
                        entry['neigh1_pos'].append([float(row['longitude']),float(row['latitude'])])
                    
                    
                    dist = compute_dist(lat2, long2, str(row['latitude']), str(row['longitude']))
                    
                    try:
                        dist = int(dist)
                    except ValueError:
                        continue
                    
                    if dist < NEIGHBORHOOD_RADIUS and len(entry['neigh2'])<50:
                        if (name1 == row['name'].lower() and str(row['latitude']) == lat1 and str(row['longitude']) == long1) or (name2 == row['name'].lower() and str(row['latitude']) == lat2 and str(row['longitude']) == long2):
                            continue
                        entry['neigh2'].append(row['name'].lower())
                        entry['neigh2_attr'].append(neighbor_attr.lower())
                        entry['dist2'].append(dist)
                        entry['neigh2_pos'].append([float(row['longitude']),float(row['latitude'])])
                        
                
                entries.append(entry)
                c+=1
                if c%100 == 0:
                    print(str(c/count*100) + '%')
                
                
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=4)

import multiprocessing
pool = multiprocessing.Pool(processes=10)
params = []
for city in ['tor','edi','sin','pit']:
    for split in ['/osm_yelp/','/osm_fsq/']:
        params.append((city,split))
pool.map(main, params)
pool.close()
pool.join()