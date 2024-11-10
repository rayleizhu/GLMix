# python main_cx2.py \
#     data_path=./data/in1k input_size=224  batch_size=128 dist_eval=true \
#     eval=true model='glnet_stl' load_release=true \

python main_tklb.py \
    data_path=./data/in1k input_size=224  batch_size=128 dist_eval=true \
    eval=true model='glnet_4g_tklb' load_release=true benchmark=false \

