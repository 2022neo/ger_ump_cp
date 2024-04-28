from easydict import EasyDict as edict
import hashlib
SEED = 123
DEVICE = 'cuda:7'
LANGUAGE_MODEL_NAME='bert-base-uncased'
sampled_prefix = 'sampled-osm_fsq-osm_yelp-NR_2000-DSPC_5000-UVP_True'

def get_sampling_config():
    sampling_config = edict(
        CITYS = ['pit','edi','sin','tor'],
        SRCS = ['osm_fsq','osm_yelp'],
        SRC_PREFIX = './formatted_data/formatted_',
        SAMPLE_ROOT = './pretrain_data/',
        PAIRED_DATASET_SIZE_PER_CITY=5000,
        RAW_NEIGHBORHOOD_RADIUS = 2000,
        USE_VALID_POS = True,
    )
    FILETAG = 'sampled-' + '-'.join(sampling_config.SRCS) + '-' + 'NR_'+str(sampling_config.RAW_NEIGHBORHOOD_RADIUS) + '-' + 'DSPC_'+str(sampling_config.PAIRED_DATASET_SIZE_PER_CITY)
    FILETAG += '-' + 'UVP_'+str(sampling_config.USE_VALID_POS)
    sampling_config.FILETAG=FILETAG
    return sampling_config
sampling_config = get_sampling_config()
print(sampling_config,'\n')

def get_loader_config():
    loader_config = edict(
        add_token=True,
        drop_null = False,
        add_padding = True,
        padding_max_length = 128,
        min_val_token_cnt = 1,
        global_mask_prob = 0.25,
        local_replace_prob = 0.1,
        neighbor_dist_thred=1000,
        e1_e2_dist_thred = 100,
        min_neighbor_num = 50,
        max_neighbor_num = 150,
        use_pre_encoder=True,
    )
    loader_citys = ['pit','edi','sin','tor']
    loader_config_tag = '-'.join(loader_citys+[f'{k}_{v}' for k,v in loader_config.items()])
    loader_config_tag =hashlib.md5(loader_config_tag.encode()).hexdigest()
    loader_dir = '/mnt/16t_3/jiyuwen/pretrain_loader'
    loader_root = f'{loader_dir}/{sampled_prefix}/{loader_config_tag}'
    loader_config.loader_root=loader_root
    loader_config.loader_citys=loader_citys
    loader_config.sampled_prefix=sampled_prefix
    loader_config.loader_dir=loader_dir
    loader_config.SAMPLE_ROOT=sampling_config.SAMPLE_ROOT
    return loader_config
loader_config=get_loader_config()
print(loader_config,'\n')


# def get_pretrain_config():
#     pretrain_config = edict(
#         pretrain_epochs = 300,
#         batch_size = 3,
#         opt_batch_count = 10,
#         node_feat_dim = 768,
#         hidd_stat_dim = 128,
#         edge_attr_dim = 1,
#         early_stop_patience = 3,
#         dropout_p = 0.4,
#         add_self_loops = True,
#         aggr = 'add',
#         use_act= True,
#         ignore_tokens = [0, 102], #[CLS]=101 [MASK]=103 [SEP]=102 [PAD]=0
#         train_ratio = 0.8,
#         penalty_ctrst = 0.1,
#         penalty_match = 0.01,
#         penalty_mlm = 0.4,
#         ctrst_tau = 0.1,      #temperature of contrastive learning
#         use_weighted_pos = True,  # only calculate loss on ego node
#         global_pool_name = 'add',  #add
#         pad_attn_scale = -10, # to ignore padding token
#         lr = 0.00001,
#     )
#     pretrain_config_tag = '-'.join([f'{k}_{"_".join([str(vi) for vi in v])}' if isinstance(v,list) else f'{k}_{v}' for k,v in pretrain_config.items()])
#     pretrain_dir = './pretrain_ckpt'
#     pretrain_config_tag=pretrain_config_tag.replace(".","d").replace(":","")
#     hash_object =hashlib.md5(pretrain_config_tag.encode()).hexdigest()
#     pretrain_ckpt_path = f'{pretrain_dir}/{sampled_prefix}/{hash_object}/best_epoch.pt'
#     pretrain_config.pretrain_ckpt_path=pretrain_ckpt_path
#     return pretrain_config
# pretrain_config = get_pretrain_config()
# print(pretrain_config,'\n')


# def get_finetune_config():
#     finetune_config = edict(
#         epochs = 10,
#         batch_size=5,
#         model='yes.pt',
#     )
#     finetune_config_tag = '-'.join([f'{k}_{v}' for k,v in finetune_config.items()])
#     finetune_dir = './finetune_ckpt'
#     pretrain_ckpt_path = f'{finetune_dir}/{sampled_prefix}/{finetune_config_tag.replace(".","d")}'
#     loader_config.pretrain_ckpt_path=pretrain_ckpt_path
#     return finetune_config


def is_valid_pos(city,lat,lon):
    if city=='sin' and (lat<1.2376256937422347 or lat>1.4708569296084157 or lon<103.61698784239873 or lon>104.03492993227736):
        return False
    if city=='pit' and (lat<40.361565999999996 or lat>40.501037 or lon<-80.09550899999999 or lon>-79.865794):
        return False
    if city=='tor' and (lat<43.58464308210616 or lat>43.855547992182615 or lon<-79.64745469478004 or lon>-79.1040939249301):
        return False
    if city=='edi' and (lat<55.88865310727855 or lat>55.99627499702918 or lon<-3.4315589422688855 or lon>-3.0613140695020777):
        return False
    return True

from math import sin, cos, sqrt, atan2, radians
def custom_distance(point1, point2):
    lat1, lon1 = point1[0],point1[1]
    lat2, lon2 = point2[0],point2[1]
    R = 6373.0
        
    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
        
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))
                
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return round(R * c * 1000)