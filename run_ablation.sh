for city in sin pit edi tor; do
    for source in osm_fsq osm_yelp; do
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 0 --global_noise_prob 0.2 --attn_type softmax --pretrain_ckpt ./save_models/attntype_softmax-epoch_latest.pth --use_geoer 1 --repeat 10 --epochs 10
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 0 --global_noise_prob 0.2 --attn_type sigmoid_relu --pretrain_ckpt ./save_models/attntype_sigmoid_relu-epoch_latest.pth --use_geoer 1  --repeat 10 --epochs 10
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 0 --global_noise_prob 0.2 --attn_type sigmoid --pretrain_ckpt ./save_models/attntype_sigmoid-epoch_latest.pth --use_geoer 1  --repeat 10 --epochs 10
    done
done
for city in sin pit edi tor; do
    for source in osm_fsq osm_yelp; do
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type softmax --pretrain_ckpt ./no/pretrain --use_geoer 1 --repeat 10 --epochs 10
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type sigmoid_relu --pretrain_ckpt ./no/pretrain --use_geoer 1  --repeat 10 --epochs 10
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type sigmoid --pretrain_ckpt ./no/pretrain --use_geoer 1  --repeat 10 --epochs 10
    done
done
for city in sin pit edi tor; do
    for source in osm_fsq osm_yelp; do
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type softmax --pretrain_ckpt ./save_models/attntype_softmax-epoch_latest.pth --use_geoer 1 --trivial_neighbor 1 --batch_size 16 --repeat 10 --epochs 10
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type sigmoid_relu --pretrain_ckpt ./save_models/attntype_sigmoid_relu-epoch_latest.pth --use_geoer 1 --trivial_neighbor 1 --batch_size 16  --repeat 10 --epochs 10
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob 0.2 --attn_type sigmoid --pretrain_ckpt ./save_models/attntype_sigmoid-epoch_latest.pth --use_geoer 1 --trivial_neighbor 1 --batch_size 16  --repeat 10 --epochs 10
    done
done
#
# sh run_ablation.sh 7 0.2