# Unaligned Message-Passing and Contextualized-Pretraining for Robust Geo-Entity Resolution
<div id="top" align="center">
<p align="center">
<img src="figures/fig_framework.png" width="1000px" >
</p>
</div>

A framework for enhancing the robustness of current geo-entity resolution methods based on pre-trained language models, accepted by [AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/view/33290). 
Geo-entity resolution involves linking records that refer to the same entities across different spatial datasets.

> 📝 **Paper**: https://ojs.aaai.org/index.php/AAAI/article/view/33290<br/>
> ✒️ **Authors**: Yuwen Ji, Wenbo Xie, Jiaqi Zhang, Chao Wang, Ning Guo, Lei Shi, Yue Zhang


### Contextualized Pretraining (CP)
To generate pretraining datasets, run:
```
python prepare_pretraining_dataset.py
```

To pre-train different versions of the framework, run:

```
python pretrain.py --device 0 --attn_type softmax ----max_epochs 20
```
The meaning of the flags and their possible values are listed here:
* ``--device``: Specify the number of which cuda you wish to use for pre-training.
* ``--attn_type``: Specify the attn_type you wish to use for message-passing. Possible values are ``softmax``, ``sigmoid``, ``sigmoid_relu``.
* ``--max_epochs``: Specify the maximum of epoch you wish to use for pre-training.

### Training (UMP)
To reproduce our main results reported in the paper.
```
sh run_ours.sh 0 0.2 && sh run_ours.sh 0 0.4
```
Running ``sh run_ours.sh 0 0.2`` means executing the main experiments with a perturbation rate of ``0.2`` using ``CUDA:0``.

### Ablation
To reproduce our ablation results reported in the paper.
```
cd dataset/
python search_neighbor_ablation.py
cd ..
sh run_ablation.sh 0 0.2
```
Running ``sh run_ablation.sh 0 0.2`` means executing the ablation experiments with a perturbation rate of ``0.2`` using ``CUDA:0``.



## Citation
```bibtex
@inproceedings{ji2025unaligned,
  title={Unaligned Message-Passing and Contextualized-Pretraining for Robust Geo-Entity Resolution},
  author={Ji, Yuwen and Xie, Wenbo and Zhang, Jiaqi and Wang, Chao and Guo, Ning and Shi, Lei and Zhang, Yue},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={11},
  pages={11852--11860},
  year={2025}
}
```
