import argparse
from functions import prepare_dataset,compute_dist,get_lat_long
from train import train_GeoER
from model import GeoER
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import json
import random

parser = argparse.ArgumentParser(description='GeoER')
parser.add_argument('-s', '--source', type=str, default='osm_fsq', help='Data source (osm_yelp, osm_fsq)')
parser.add_argument('-c', '--city', type=str, default='pit', help='Data source (sin, edi, tor, pit)')

args = parser.parse_args()

root_path = "./data/train_valid_test/"
root_n_path = "./data/ext_neighborhood_train_valid_test/"
out_path = "./dataset2/train_valid_test/"
out_n_path = "./dataset2/ext_neighborhood_train_valid_test/"

def trans_dataset(path_list, n_path_list):
    data_positive = []
    data_negative = []
    max_coord = 0
    for path,n_path in zip(path_list, n_path_list):
        with open(n_path, 'r', encoding='utf-8') as f:
            neigh = json.load(f)
        with open(path, 'r') as f:
            for line,nei in zip(f,neigh):
                arr = line.split('\t')
                e1, lat1, long1 = get_lat_long(arr[0])
                e2, lat2, long2 = get_lat_long(arr[1])
                coord = compute_dist(lat1, long1, lat2, long2)
                max_coord = max(max_coord,coord)
                y = int(line.split('\t')[2].strip())
                if y ==1:
                    data_positive.append((nei,line,coord))
                elif y==0:
                    data_negative.append((nei,line,coord))
    trainset,testset = [],[]
    
    data_negative = sorted(data_negative, key=lambda x: x[-1])
    data_negative = data_negative[:len(data_positive)]
    random.shuffle(data_negative)
    random.shuffle(data_negative)
    split_point = 727
    trainset += data_negative[:split_point]
    testset += data_negative[split_point:]

    random.shuffle(data_positive)
    random.shuffle(data_positive)
    split_point = 727
    trainset += data_positive[:split_point]
    testset += data_positive[split_point:]

    random.shuffle(trainset)
    random.shuffle(testset)
    random.shuffle(trainset)
    random.shuffle(testset)

    validset = trainset[:454]
    trainset = trainset[454:]
    return trainset,validset,testset



def get_dataset(args):
    train_path = root_path + args.source + '/' + args.city + '/train.txt'
    n_train_path = root_n_path + args.source + '/' + args.city + '/n_train.json'

    valid_path = root_path + args.source + '/' + args.city + '/valid.txt'
    n_valid_path = root_n_path + args.source + '/' + args.city + '/n_valid.json'

    test_path = root_path + args.source + '/' + args.city + '/test.txt'
    n_test_path = root_n_path + args.source + '/' + args.city + '/n_test.json'

    path_list = [train_path,valid_path,test_path]
    n_path_list = [n_train_path,n_valid_path,n_test_path]
    trainset,validset,testset = trans_dataset(path_list, n_path_list)
    print(f'trainset:{len(trainset)}, validset:{len(validset)}, testset:{len(testset)}')


    out_train_path = out_path + args.source + '/' + args.city + '/train.txt'
    out_n_train_path = out_n_path + args.source + '/' + args.city + '/n_train.json'

    out_valid_path = out_path + args.source + '/' + args.city + '/valid.txt'
    out_n_valid_path = out_n_path + args.source + '/' + args.city + '/n_valid.json'

    out_test_path = out_path + args.source + '/' + args.city + '/test.txt'
    out_n_test_path = out_n_path + args.source + '/' + args.city + '/n_test.json'

    Path(out_train_path).parent.mkdir(exist_ok=True,parents=True)
    Path(out_n_train_path).parent.mkdir(exist_ok=True,parents=True)
    Path(out_valid_path).parent.mkdir(exist_ok=True,parents=True)
    Path(out_n_valid_path).parent.mkdir(exist_ok=True,parents=True)
    Path(out_test_path).parent.mkdir(exist_ok=True,parents=True)
    Path(out_n_test_path).parent.mkdir(exist_ok=True,parents=True)

    neigh_train = []
    with open(out_train_path,'w', encoding='utf-8') as f:
        for nei,line,coord in trainset:
            neigh_train.append(nei)
            f.write(line)
    with open(out_n_train_path,'w', encoding='utf-8') as f:
        json.dump(neigh_train,f, indent=4)

    neigh_valid = []
    with open(out_valid_path,'w', encoding='utf-8') as f:
        for nei,line,coord in validset:
            neigh_valid.append(nei)
            f.write(line)
    with open(out_n_valid_path,'w', encoding='utf-8') as f:
        json.dump(neigh_valid,f, indent=4)

    neigh_test = []
    with open(out_test_path,'w', encoding='utf-8') as f:
        for nei,line,coord in testset:
            neigh_test.append(nei)
            f.write(line)
    with open(out_n_test_path,'w', encoding='utf-8') as f:
        json.dump(neigh_test,f, indent=4)

get_dataset(args)
