#!/bin/bash

# -- Setting --
gpu=0

# teacher='wrn_40_2'
teacher='resnet32x4'
# student='wrn_40_1'
# student='ShuffleV2' 
student='resnet8x4' 
# student='wrn_16_2'
# method='cifar100_lr_0.05_decay_0.0005'
# method='cifar100_lr_0.05_decay_0.0005_swa=150'
# method='cifar100_lr_0.05_decay_0.0005_curl_alpha=1_linear_swa=150'
# method='cifar100_lr_0.05_decay_0.0005_swareg_alpha=1_linear'
method='cifar100_lr_0.05_decay_0.0005_swareg_alpha=1_linear_swa=150'
trial=0




# -- Config --
cur=$PWD
cd CIFAR-100
python config_distillation.py --teacher_path "$cur"/experiments/models/"$teacher"_"$method"_trial_"$trial"/"$teacher"_last.pth --model_s "$student" --gpu "$gpu"
[[ $? -ne 0 ]] && echo 'exit' && exit 2

# -- RUN --
path=$(cat ./tmp/config.tmp | grep 'exp_path' | awk '{print$NF}' | tr -d '"')
cp ./tmp/config.tmp $path
python train_with_distillation.py --config_file "$path/config.tmp" > $path/train.out 2>&1 &

pid=$!
echo "[$pid] [$gpu] [Path]: $path"
echo "s [$pid] [$gpu] $(date) [Path]: $path" >> ./logs/log.txt
cd $cur

