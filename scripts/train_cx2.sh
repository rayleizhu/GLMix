
set -x

models=(
    "glnet_4g"
    "glnet_9g"
)

# by default, eff_bs = slurm.nodes * 8 * batch_size * update_freq = 2048, lr=2e-3
for model in ${models[@]}; do
    python main_cx2.py \
        model=${model} drop_path=0.1 lr=2e-3 clip_grad=5.0 model_ema_decay=0.9999 \
        input_size=224 use_prefetch=true \
        +slurm=${CLUSTER_ID} slurm.nodes=2 batch_size=128 slurm.quotatype='spot'
        # +pavi=default pavi.project='glnet.cx2script'
    sleep 1
done

