set -e
for city in sin pit edi tor; do
    for source in osm_fsq osm_yelp; do
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --onlyname 0 --global_noise_prob $2 --attn_type $3 --pretrain_ckpt $4 --use_neighbor 1 --repeat 3
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --onlyname 0 --global_noise_prob $2 --attn_type $3 --pretrain_ckpt $4 --use_neighbor 0 --repeat 3
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --onlyname 1 --global_noise_prob $2 --attn_type $3 --pretrain_ckpt $4 --use_neighbor 1 --repeat 3
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --onlyname 1 --global_noise_prob $2 --attn_type $3 --pretrain_ckpt $4 --use_neighbor 0 --repeat 3
    done
done
# sh run_ump_ft.sh 6 1.0 softmax ./save_models/attntype_softmax-epoch_16-loss_8629.pth
# sh run_ump_ft.sh 6 1.0 sigmoid_relu ./save_models/attntype_sigmoid_relu-epoch_20-loss_11787.pth
# sh run_ump_ft.sh 8 0.0 sigmoid ./save_models/attntype_sigmoid-epoch_20-loss_13116.pth
# sh run_ump_ft.sh 7 0.8 softmax /no/pretrain/
# sh run_ump_ft.sh 8 0.8 sigmoid_relu /no/pretrain/
# sh run_ump_ft.sh 9 0.8 sigmoid /no/pretrain/