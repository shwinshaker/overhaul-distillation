
paths=(

    # 'wrn_40_2_cifar100_lr_0.05_decay_0.0005_swarank_alpha=1_linear_swa=150_trial_0'
    'resnet32x4_cifar100_lr_0.05_decay_0.0005_swarank_alpha=1_linear_swa=150_trial_0'

)



sub_dir='models'
dir_src='../RepDistiller/save'
dir_dest='experiments'

for path in "${paths[@]}"; do
    echo
    echo '==>' $path
    cp -rs "$(readlink -f $dir_src)/$sub_dir/$path" "$dir_dest/$sub_dir"
done
