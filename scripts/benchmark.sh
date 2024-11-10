

models=(
    'glnet_4g'
    'glnet_9g'
    'glnet_16g'
    'glnet_4g_tklb'
    'glnet_9g_tklb'
    'glnet_stl'
    'glnet_stl_paramslot'
)

torch_version=$(python -c "import torch; print(torch.__version__)")
precision=float32
batch_size=128
sdpa_kernel=math
results_file=torch_version.${torch_version}-spda_kernel.${sdpa_kernel}-bs.${batch_size}-precision.${precision}.csv

now=$(date '+%y%m%d-%H.%M.%S')
workdir=outputs/benchmark/${now}
mkdir -p ${workdir}
printf "%s\n" ${models[@]} >> ${workdir}/model_list.txt

# srun -p mediaa --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 \
python benchmark.py \
    --bench inference \
    --model-list ${workdir}/model_list.txt \
    --results-file ${workdir}/${results_file} \
    --sdpa-kernel ${sdpa_kernel} \
    --batch-size ${batch_size} \
    --precision ${precision}









