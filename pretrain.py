from config_pretrain import loader_config
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from difflib import SequenceMatcher
import json
from tqdm import tqdm
import torch
import random
import torch.optim as optim
from pathlib import Path
from copy import deepcopy
import argparse
import sys
sys.path.append('../')
from model import GeoUmpForPretrain
import config

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
cls_id = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
sep_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
pad_id = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
col_id,val_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('COL VAL'))

parser = argparse.ArgumentParser(description='Pretrain')
parser.add_argument('--attn_type', type=str, default='sigmoid',help='sigmoid_relu sigmoid softmax ')
parser.add_argument('-d','--device', type=int, default=7)
args = parser.parse_args()
config.attn_type = args.attn_type
config.device = f'cuda:{args.device}'
device = f'cuda:{args.device}'

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
    
    
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def perturb_tokens(sentence):
    tokens = deepcopy(sentence)
    skip_token_list = [101,102,103,0,8902,11748]
    for i in range(0,len(tokens)):
        if tokens[i] in skip_token_list:
            sentence[i]=-100
            continue
        else:
            prob = 0.3
        if np.random.rand()<prob:
            if np.random.rand() < 0.5:
                tokens[i]=np.random.randint(tokenizer.vocab_size)
            else:
                tokens[i]=103 #mask_id
        else:
            if np.random.rand() < 0.5:
                sentence[i]=-100
    return tokens
def entity_to_token(row):
    orig_x = []
    for k,v in row.items():
        if k=='longitude' or k == 'latitude':
            continue
        if not isinstance(v,str):
            print(k,v,type(v),bool(v))
        key_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(k))
        val_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(v))
        orig_x += [col_id] + key_tokens + [val_id] + val_tokens
    return orig_x,row['name']

def get_dataset(city='pit'):
    sampled_csv_p = f'{loader_config.SAMPLE_ROOT}/{loader_config.sampled_prefix}/{city}.csv'
    entity_df_dict = {}
    for entity_src in ['osm_yelp','osm_fsq']:
        entity_table_p = f'{loader_config.SAMPLE_ROOT}/{entity_src}/{city}.csv'
        entity_df_dict[entity_src]=pd.read_csv(entity_table_p,index_col=0).fillna(' ')
    data_df = pd.read_csv(sampled_csv_p, index_col=0)
    data1 = []
    data2 = []
    for ind, row in data_df.iterrows():
        e1_id,e1n_ids,e1n_dst,e1_tb,e2_id,e2n_ids,e2n_dst,e2_tb,dst_e1_e2=row
        if dst_e1_e2 > loader_config.e1_e2_dist_thred:
            continue
        token1,name1 = entity_to_token(entity_df_dict[e1_tb].iloc[e1_id])
        token2,name2 = entity_to_token(entity_df_dict[e2_tb].iloc[e2_id])

        neigh_list = []
        coord_list = []
        neighbor_num = np.random.randint(loader_config.min_neighbor_num,loader_config.max_neighbor_num)
        for en_id, dst in zip(json.loads(e1n_ids),json.loads(e1n_dst)):
            if len(neigh_list)>neighbor_num:
                break
            if dst > loader_config.neighbor_dist_thred:
                continue
            nei,name = entity_to_token(entity_df_dict[e1_tb].iloc[en_id])
            # if not ((jaccard_similarity(name1.split(), name.lower().split()) > 0.4 or similar(name1, name.lower()) > 0.75)):
            #     continue
            neigh_list.append(nei)
            coord_list.append(dst)
        data1.append(
            (token1,neigh_list,coord_list)
        )

        neigh_list = []
        coord_list = []
        neighbor_num = np.random.randint(loader_config.min_neighbor_num,loader_config.max_neighbor_num)
        for en_id, dst in zip(json.loads(e2n_ids),json.loads(e2n_dst)):
            if len(neigh_list)>neighbor_num:
                break
            if dst > loader_config.neighbor_dist_thred:
                continue
            nei,name = entity_to_token(entity_df_dict[e2_tb].iloc[en_id])
            # if not ((jaccard_similarity(name2.split(), name.lower().split()) > 0.2 or similar(name2, name.lower()) > 0.4)):
            #     continue
            neigh_list.append(nei)
            coord_list.append(dst)
        data2.append(
            (token2,neigh_list,coord_list)
        )
        # if len(data2)>60:
        #     break
    dataset = []
    sampled_inds = torch.randperm(len(data1))
    for ind in range(0,len(sampled_inds),2):
        if ind+1<len(sampled_inds):
            dataset.append(
                (data1[sampled_inds[ind]],data2[sampled_inds[ind+1]],0)
            )
    sampled_inds = torch.randperm(len(data1))
    for ind in range(0,len(sampled_inds),2):
        if ind+1<len(sampled_inds):
            dataset.append(
                (data2[sampled_inds[ind]],data1[sampled_inds[ind+1]],0)
            )
    sampled_inds = torch.randperm(len(data1))
    for ind in range(0,len(sampled_inds),1):
        dataset.append(
            (data1[sampled_inds[ind]],data2[sampled_inds[ind]],1)
        )
    random.shuffle(dataset)
    random.shuffle(dataset)
    return [e[0] for e in dataset],[e[1] for e in dataset],[e[2] for e in dataset]



def get_model(device):
    model = GeoUmpForPretrain(device=device, n_emb=config.n_em, a_emb=config.a_em, dropout=config.dropout)
    model = model.to(device)
    save_model_path = f'./save_models/attntype_{model.ump_layer.attn_type}-epoch_latest.pth'
    print(save_model_path)
    pretrain_log=None
    if Path(save_model_path).exists():
        ckpt = torch.load(save_model_path,map_location=device)
        model.load_state_dict(ckpt['pretrain_state_dict'])
        pretrain_log = ckpt['pretrain_log']
    else:
        Path(save_model_path).parent.mkdir(exist_ok=True,parents=True)
    return model,pretrain_log,save_model_path

model,pretrain_log,save_model_path = get_model(device)
citys = ['pit','edi','sin','tor']
if pretrain_log:
    start_epoch = pretrain_log['epoch']+1
    epoch_lr = pretrain_log['epoch_lr']
    weight_decay  = pretrain_log['weight_decay']
    epoch_lr = 1e-5
    weight_decay = 0.0
    best_train_loss = pretrain_log['best_train_loss']
else:
    start_epoch = 1
    epoch_lr = 1e-5
    weight_decay = 0.0
    best_train_loss = float('inf')

optimizer = optim.Adam(model.parameters(), lr=epoch_lr, weight_decay=weight_decay)
model.train()
for epoch in range(start_epoch,100):
    train_loss=0
    city = citys[epoch%4]
    data1,data2,labels=get_dataset(city)
    # data1,data2,labels = data1[:4],data2[:4],labels[:4]
    i=0
    batch_size = 2
    masked_sentences,x_n,e12pos_list,pair_labels,token_labels = [],[],[],[],[]
    pbar = tqdm(total=len(data1), desc=f"Epoch {epoch} Pretrain...")
    for (token1,neigh1_list,dist1),(token2,neigh2_list,dist2),label in zip(data1,data2,labels):
        i+=1
        sentence = [cls_id] + token1 + [sep_id] + token2 + [sep_id]
        e1_pos = [1,len(token1)]
        e2_pos = [len(token1)+2,len(token1)+len(token2)+1]
        if len(sentence) < 128:
            sentence += [pad_id]*(128 - len(sentence))
        else:
            sentence = sentence[:128]
        masked_sentences.append(perturb_tokens(sentence))
        x_n.append({
            'neigh1_attr':neigh1_list,
            'neigh2_attr':neigh2_list,
            'dist1':dist1,
            'dist2':dist2,
        })
        e12pos_list.append(
            (e1_pos,e2_pos)
        )
        pair_labels.append(label)
        token_labels.append(sentence)

        if i%batch_size==0 or i==len(data1):
            pair_labels = torch.tensor(pair_labels).to(device)
            token_labels = torch.tensor(token_labels).to(device)
            x = torch.tensor(masked_sentences)
            if len(x.shape) < 2:
                x = x.unsqueeze(0)
            if len(token_labels.shape) < 2:
                token_labels = token_labels.unsqueeze(0)
            att_mask = torch.tensor(np.where(x != 0, 1, 0)).to(device)
            x = x.to(device)
            bs = batch_size if i%batch_size==0 else i%batch_size
            loss = model(x, x_n, att_mask, bs, e12pos_list,token_labels,pair_labels)
            loss.backward()
            if i%32 == 0:
                optimizer.step()
                optimizer.zero_grad()
            train_loss +=loss.item()
            pbar.set_postfix_str(f'train_loss={train_loss}')
            masked_sentences,x_n,e12pos_list,pair_labels,token_labels = [],[],[],[],[]
        pbar.update(1)
    pbar.close()
    optimizer.step()
    optimizer.zero_grad()
    if train_loss < best_train_loss:
        best_train_loss = train_loss
    else:
        epoch_lr *= 0.9
        weight_decay += 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = epoch_lr
            param_group['weight_decay'] = weight_decay
    epoch_log = {
        'epoch':epoch,
        'epoch_lr':epoch_lr,
        'weight_decay':weight_decay,
        'best_train_loss':train_loss,
    }
    epoch_ckpt = {
        'pretrain_state_dict':model.state_dict(),
        'language_model':model.language_model.state_dict(),
        'neighbert':model.neighbert.state_dict(),
        'ump_layer':model.ump_layer.state_dict(),
        'pretrain_log':epoch_log,
    }
    epoch_model_path = save_model_path.replace('epoch_latest',f'epoch_{epoch}-loss_{int(train_loss)}')
    torch.save(epoch_ckpt,save_model_path)
    torch.save(epoch_ckpt,epoch_model_path)
