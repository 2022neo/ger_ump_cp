MATCH = 1
NO_MATCH = 0
MAX_DIST = 2000
lm_hidden = 768
c_em = 256
n_em = 256
a_em = 256
max_seq_len = 128
device = 'cuda:7'
dropout = 0.2
epochs = 10
batch_size = 32
lr=3e-5

add_noise=True
use_ump=True
use_geoer=True
global_noise_prob = 0.2
attn_type = 'sigmoid'  #sigmoid_relu sigmoid softmax

repeat = 3
path = "./dataset/train_valid_test/"
n_path = "./dataset/ext_neighborhood_train_valid_test/"
log_path = "./log_dir/"
seed = 123