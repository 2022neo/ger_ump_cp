for city in sin pit edi tor; do
    for source in osm_fsq osm_yelp; do
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob $2 --attn_type softmax --pretrain_ckpt ./save_models/attntype_softmax-epoch_16-loss_8629.pth --use_geoer 0  --repeat 10 --epochs 10
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob $2 --attn_type sigmoid_relu --pretrain_ckpt ./save_models/attntype_sigmoid_relu-epoch_20-loss_11787.pth --use_geoer 0  --repeat 10 --epochs 10
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob $2 --attn_type sigmoid --pretrain_ckpt ./save_models/attntype_sigmoid-epoch_20-loss_13116.pth --use_geoer 0  --repeat 10 --epochs 10
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob $2 --attn_type softmax --pretrain_ckpt ./save_models/attntype_softmax-epoch_16-loss_8629.pth --use_geoer 1 --repeat 10 --epochs 10
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob $2 --attn_type sigmoid_relu --pretrain_ckpt ./save_models/attntype_sigmoid_relu-epoch_20-loss_11787.pth --use_geoer 1  --repeat 10 --epochs 10
        python main_src.py -c "$city" -s "$source" -d $1 --use_ump 1 --global_noise_prob $2 --attn_type sigmoid --pretrain_ckpt ./save_models/attntype_sigmoid-epoch_20-loss_13116.pth --use_geoer 1  --repeat 10 --epochs 10
    done
done
#
# sh run_ours.sh 5 0.2 && sh run_ours.sh 6 0.4