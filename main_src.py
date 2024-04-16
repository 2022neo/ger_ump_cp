import argparse
import config
from functions import prepare_dataset
from train import train_GeoER
from model import GeoUmpER
import numpy as np
from pathlib import Path
import json
import torch

parser = argparse.ArgumentParser(description='GeoER')
parser.add_argument('-s', '--source', type=str, default='osm_fsq', help='Data source (osm_yelp, osm_fsq)')
parser.add_argument('-c', '--city', type=str, default='pit', help='Data source (sin, edi, tor, pit)')

parser.add_argument('--add_noise', type=int, default=1)
parser.add_argument('--use_ump', type=int, default=1)
parser.add_argument('--pretrain_ckpt', type=str, default='path/to/ckpt')
parser.add_argument('--onlyname', type=int, default=1)
parser.add_argument('--add_neighbor_noise', type=int, default=1)
parser.add_argument('--use_neighbor', type=int, default=1)
parser.add_argument('--global_noise_prob', type=float, default=0.2)
parser.add_argument('-d','--device', type=int, default=7)
parser.add_argument('-i','--inject_sep', type=int, default=1)
parser.add_argument('--attn_type', type=str, default='sigmoid_relu')
parser.add_argument('--repeat', type=int, default=3)
parser.add_argument('--ablation', type=int, default=0)

# parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--lr', type=float, default=3e-5)
# parser.add_argument('--c_em', type=int, default=256)
# parser.add_argument('--n_em', type=int, default=256)
# parser.add_argument('--a_em', type=int, default=256)


args = parser.parse_args()


citys = [args.city]
# citys = ['sin', 'edi', 'tor', 'pit']

config.repeat = args.repeat
config.add_noise = bool(args.add_noise)
config.use_ump = bool(args.use_ump)
config.onlyname = bool(args.onlyname)
config.add_neighbor_noise = bool(args.add_neighbor_noise)
config.use_neighbor = bool(args.use_neighbor)
config.global_noise_prob = args.global_noise_prob
config.device = f'cuda:{args.device}'
config.attn_type = args.attn_type
config.inject_sep = bool(args.inject_sep)
config.n_path = "./dataset/ablation_neighborhood_train_valid_test/" if args.ablation else config.n_path
# config.dropout = args.dropout
config.batch_size = args.batch_size
# config.lr = args.lr
# config.c_em = args.c_em
# config.n_em = args.n_em
# config.a_em = args.a_em

if args.ablation:
    log_path = Path(config.log_path+'_ablation')/f"noise_{str(config.global_noise_prob).replace('.','d')}"/f"{args.source.lower()+'_'+'_'.join(citys) + '.log'}"
else:
    log_path = Path(config.log_path)/f"noise_{str(config.global_noise_prob).replace('.','d')}"/f"{args.source.lower()+'_'+'_'.join(citys) + '.log'}"
log_path = str(log_path)
print(log_path)
if not Path(log_path).exists():
    Path(log_path).parent.mkdir(exist_ok=True,parents=True)

device = config.device
def get_dataset(args,config):
    all_train_x, all_train_coord, all_train_n, all_train_y = [],[],[],[]
    all_valid_x, all_valid_coord, all_valid_n, all_valid_y = [],[],[],[]
    all_test_x, all_test_coord, all_test_n, all_test_y = [],[],[],[]
    for city in citys:
        train_path = config.path + args.source + '/' + city + '/train.txt'
        n_train_path = config.n_path + args.source + '/' + city + '/n_train.json'
        train_x, train_coord, train_n, train_y = prepare_dataset(train_path, n_train_path, max_seq_len=config.max_seq_len)
        all_train_x+=train_x
        all_train_coord+=train_coord
        all_train_n+=train_n
        all_train_y+=train_y


        valid_path = config.path + args.source + '/' + city + '/valid.txt'
        n_valid_path = config.n_path + args.source + '/' + city + '/n_valid.json'
        valid_x, valid_coord, valid_n, valid_y = prepare_dataset(valid_path, n_valid_path, max_seq_len=config.max_seq_len)
        all_valid_x += valid_x
        all_valid_coord += valid_coord
        all_valid_n += valid_n
        all_valid_y += valid_y

        test_path = config.path + args.source + '/' + city + '/test.txt'
        n_test_path = config.n_path + args.source + '/' + city + '/n_test.json'
        test_x, test_coord, test_n, test_y = prepare_dataset(test_path, n_test_path, max_seq_len=config.max_seq_len)
        all_test_x += test_x
        all_test_coord += test_coord
        all_test_n += test_n
        all_test_y += test_y

    if args.ablation:
        new_ns = [[],[],[]]
        for ind,x_n in enumerate([all_train_n,all_valid_n,all_test_n]):
            for b in range(len(x_n)):
                new_ns[ind].append(
                    {
                        'neigh1' : x_n[b]['neigh1'][:100],
                        'neigh2' : x_n[b]['neigh2'][:100],
                        'dist1' : x_n[b]['dist1'][:100],
                        'dist2' : x_n[b]['dist2'][:100],
                        'name1' : x_n[b]['name1'][:100],
                        'name2' : x_n[b]['name2'][:100],
                        'neigh1_attr' : x_n[b]['neigh1_attr'][:100],
                        'neigh2_attr' : x_n[b]['neigh2_attr'][:100],
                    }
                )
                # x_n[b]['neigh1'] = x_n[b]['neigh1'][:100]
                # x_n[b]['neigh2'] = x_n[b]['neigh2'][:100]
                # x_n[b]['dist1'] = x_n[b]['dist1'][:100]
                # x_n[b]['dist2'] = x_n[b]['dist2'][:100]
                # x_n[b]['name1'] = x_n[b]['name1'][:100]
                # x_n[b]['name2'] = x_n[b]['name2'][:100]
                # x_n[b]['neigh1_attr'] = x_n[b]['neigh1_attr'][:100]
                # x_n[b]['neigh2_attr'] = x_n[b]['neigh2_attr'][:100]
        all_train_n,all_valid_n,all_test_n = new_ns
    return all_train_x, all_train_coord, all_train_n, all_train_y, all_valid_x, all_valid_coord, all_valid_n, all_valid_y, all_test_x, all_test_coord, all_test_n, all_test_y

train_x, train_coord, train_n, train_y, valid_x, valid_coord, valid_n, valid_y, test_x, test_coord, test_n, test_y = get_dataset(args,config)

print('Succesfully loaded','(',args.source,') dataset: '+str(citys))
print('Train size:',len(train_x))
print('Valid size:',len(valid_x))


def get_model():
    model = GeoUmpER(device=device, dropout=config.dropout, c_emb=config.c_em, n_emb=config.n_em, a_emb=config.a_em)
    model = model.to(device)
    if Path(args.pretrain_ckpt).exists() and config.use_ump:
        print(f'######## load pretrain from {args.pretrain_ckpt} ###########')
        ckpt = torch.load(args.pretrain_ckpt,map_location=device)
        model.language_model.load_state_dict(ckpt['language_model'])
        model.neighbert.load_state_dict(ckpt['neighbert'])
        model.ump_layer.load_state_dict(ckpt['ump_layer'])
    return model
model = get_model()

test_f1_list = []
valid_f1_list = []

for round in range(1,config.repeat+1):
    print(f'*** Round {round}')
    test_f1,test_acc,valid_f1,valid_acc = train_GeoER(model, train_x, train_coord, train_n, train_y, valid_x, valid_coord, valid_n, valid_y, test_x, test_coord, test_n, test_y, device, epochs=config.epochs, batch_size=config.batch_size, lr=config.lr)
    test_f1_list.append(test_f1)
    valid_f1_list.append(valid_f1)


res_info = {
    'Test_F1': f'{np.mean(test_f1_list):.3f}$\pm${np.std(test_f1_list):.3f}',
    'Valid_F1': f'{np.mean(valid_f1_list):.3f}$\pm${np.std(valid_f1_list):.3f}',
    'config':{
        'add_noise':config.add_noise,
        'use_ump':config.use_ump,
        'onlyname':config.onlyname,
        'add_neighbor_noise':config.add_neighbor_noise,
        'global_noise_prob':config.global_noise_prob,
        'use_neighbor':config.use_neighbor,
        'dropout':config.dropout,
        'batch_size':config.batch_size,
        'lr':config.lr,
        'c_em':config.c_em,
        'n_em':config.n_em,
        'a_em':config.a_em,
        'Repeat':config.repeat,
        'n_path':config.n_path,
        'Epochs': config.epochs,
    }
}
tag = int(config.use_ump)
if config.use_ump:
    res_info['config']['attn_type']=config.attn_type
    if config.inject_sep:
        tag+=1
        res_info['config']['inject_sep']=config.inject_sep
        

with open(log_path,'a') as f:
    f.write(f'{tag}')
    if args.ablation:
         f.write(f'-ablation')
    if Path(args.pretrain_ckpt).exists() and config.use_ump:
         f.write(f'-{Path(args.pretrain_ckpt).stem}')
    if not config.use_neighbor:
        f.write(f'-bert_only')
    f.write(json.dumps(res_info))
    f.write('\n')
