#!/bin/bash

# -- Setting --
gpu=7

teacher='wrn_40_2'
# teacher='resnet32x4'
student='wrn_16_2'
# student='wrn_40_1'
# student='ShuffleV2' 
# student='resnet8x4' 
# method='cifar100_lr_0.05_decay_0.0005'
# method='cifar100_lr_0.05_decay_0.0005_lip_alpha=1e-05'
# method='cifar100_lr_0.05_decay_0.0005_omse_alpha=1_linear'
method='cifar100_lr_0.05_decay_0.0005_lip_alpha=1e-05_omse_alpha=1_linear'
# method='cifar100_lr_0.05_decay_0.0005_swa=150'
# method='cifar100_lr_0.05_decay_0.0005_crl_alpha=1_linear_swa=150'
# method='cifar100_lr_0.05_decay_0.0005_curl_alpha=1_linear_swa=150'
# method='cifar100_lr_0.05_decay_0.0005_swareg_alpha=1_linear'
# method='cifar100_lr_0.05_decay_0.0005_swareg_alpha=1_linear_swa=150'
# method='cifar100_lr_0.05_decay_0.0005_swarank_alpha=1_linear_swa=150'
trial=0

gamma=1
alpha=1.0 # 1.0
beta=5000


# -- Config --
cur=$PWD
cd CIFAR-100
python config_distillation.py --teacher_path "$cur"/save/models/"$teacher"_"$method"_trial_"$trial"/"$teacher"_last.pth --model_s "$student" --gpu "$gpu" --beta "$beta" --alpha "$alpha" --gamma "$gamma"
[[ $? -ne 0 ]] && echo 'exit' && exit 2

# -- RUN --
path=$(cat ./tmp/config.tmp | grep 'exp_path' | awk '{print$NF}' | tr -d '"')
cp ./tmp/config.tmp $path
python train_with_distillation.py --config_file "$path/config.tmp" > $path/train.out 2>&1 &

pid=$!
echo "[$pid] [$gpu] [Path]: $path"
echo "s [$pid] [$gpu] $(date) [Path]: $path" >> ./logs/log.txt
cd $cur

