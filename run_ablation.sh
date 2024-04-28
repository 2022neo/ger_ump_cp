# for city in sin pit edi tor; do
#     for source in osm_fsq osm_yelp; do
#         python main_src.py -c "$city" -s "$source" -d $1 --use_ump 0 --global_noise_prob 0.2 --attn_type softmax --pretrain_ckpt ./save_models/attntype_softmax-epoch_16-loss_8629.pth --use_geoer 1 --repeat 1 --epochs 1
#         python main_src.py -c "$city" -s "$source" -d $1 --use_ump 0 --global_noise_prob 0.2 --attn_type sigmoid_relu --pretrain_ckpt ./save_models/attntype_sigmoid_relu-epoch_20-loss_11787.pth --use_geoer 1  --repeat 1 --epochs 1
#         python main_src.py -c "$city" -s "$source" -d $1 --use_ump 0 --global_noise_prob 0.2 --attn_type sigmoid --pretrain_ckpt ./save_models/attntype_sigmoid-epoch_20-loss_13116.pth --use_geoer 1  --repeat 1 --epochs 1
#     done
# done
# for city in sin pit edi tor; do
#     for source in osm_fsq osm_yelp; do
#         python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type softmax --pretrain_ckpt ./no/pretrain --use_geoer 1 --repeat 1 --epochs 1
#         python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type sigmoid_relu --pretrain_ckpt ./no/pretrain --use_geoer 1  --repeat 1 --epochs 1
#         python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type sigmoid --pretrain_ckpt ./no/pretrain --use_geoer 1  --repeat 1 --epochs 1
#     done
# done
for city in sin pit edi tor; do
    for source in osm_fsq osm_yelp; do
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type softmax --pretrain_ckpt ./save_models/attntype_softmax-epoch_16-loss_8629.pth --use_geoer 1 --trivial_neighbor 1 --batch_size 16 --repeat 1 --epochs 1
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type sigmoid_relu --pretrain_ckpt ./save_models/attntype_sigmoid_relu-epoch_20-loss_11787.pth --use_geoer 1 --trivial_neighbor 1 --batch_size 16  --repeat 1 --epochs 1
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type sigmoid --pretrain_ckpt ./save_models/attntype_sigmoid-epoch_20-loss_13116.pth --use_geoer 1 --trivial_neighbor 1 --batch_size 16  --repeat 1 --epochs 1
    done
done
#
# sh run_ablation.sh 7 0.2