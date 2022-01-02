
paths=(

)



sub_dir='student_model'
dir_src='../RepDistiller/save'
dir_dest='save'

for path in "${paths[@]}"; do
    echo
    echo '==>' $path
    cp -rs "$(readlink -f $dir_src)/$sub_dir/$path" "$dir_dest/$sub_dir"
done
