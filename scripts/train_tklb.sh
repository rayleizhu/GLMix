
set -x

models=(
   'glnet_4g_tklb'
   'glnet_9g_tklb'
)

# Note that, in main_tklb.py, the lr is linearly scaled:
# linear_scaled_lr = args.lr * args.batch_size * cx2utils.get_world_size() / 512.0 
# by default, eff_bs = slurm.nodes*8*batch_size = 2048, lr=5e-4*2048/512=2e-3
for model in ${models[@]}; do
    python main_tklb.py \
        model=${model} drop_path=0.1 lr=5e-4 clip_grad=5.0 epochs=310 \
        +slurm=${CLUSTER_ID} slurm.nodes=2 batch_size=128 slurm.quotatype='spot' \
        # +pavi=default pavi.project='glnet.tklbscript' 
    sleep 1
done
