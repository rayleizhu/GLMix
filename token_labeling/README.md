

Based on https://github.com/YehLi/ImageNetModel/tree/main/classification


Below is the script I used train biformer with token labeling

```bash
time_stamp=$(date '+%m-%d-%H:%M:%S')


##### config area ########
# models=( biformerv9_CCGG_small )
models=(
        biformer_tiny_tl
        #  biformer_small_tl
        # biformer_base_tl
       )
epochs=310
nodes=1
drop_paths=( 0.1 ) # swin:0.2, uniformer 0.1
lrs=( 5e-4 ) # follows uniformer/swin
size=224
##########################

# do not touch
# update_freq=$(( 8 / ${nodes} ))
##########################

for drop_path in ${drop_paths[*]}; do
    for lr in ${lrs[*]}; do
        for model in ${models[*]}; do
            job_name=${model}.size_${size}.ep_${epochs}.dp_${drop_path}.lr_${lr}
            work_dir=local_work_dirs/${job_name}/${time_stamp}
            echo "results will be saved to ${work_dir}"

            # create dir and run task
            mkdir -p ${work_dir}
            python run_with_submitit.py --nodes ${nodes} --ngpus 8 \
                                        --model ${model} --drop-path ${drop_path} \
                                        --batch-size 128 --lr ${lr} \
                                        --input-size ${size} \
                                        --epochs ${epochs} \
                                        --token-label \
                                        --token-label-size 7 \
                                        --token-label-data /mnt/lustre/share_data/zhulei1/token_labeling/label_top5_train_nfnet \
                                        --dist-eval \
                                        --partition mediasuper \
                                        --data-path /mnt/lustre/share_data/zhulei1/ImageNet/ \
                                        --output_dir ${work_dir} \
                                        --job_dir ${work_dir} \
                                        --job_name ${job_name}
                                        # --exclude_node SZ-IDC1-10-112-2-10
        done
    done
done
```