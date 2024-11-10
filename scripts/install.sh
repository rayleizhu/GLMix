

# BUG: be careful to use conda and pip together. I found sometimes it causes extremely slow training which may be due to compatiability problem.
# install pytorch
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y && \
# conda install  -c conda-forge fvcore fairscale hydra-core einops tensorboard submitit timm scipy iopath -y
# pip --no-cache-dir install fairscale==0.4.13 fvcore==0.1.5.post20221221 hydra-core==1.3.2 einops==0.6.0 timm submitit==1.4.5 tensorboard==2.10.0

# mm series is not necessary for in1k experiments, and sometimes inatllation experience of mm series sucks.
# In that scenario, you may install it in a seperate env for downsteam tasks
pip3 install torch torchvision torchaudio \
    fairscale==0.4.13 fvcore==0.1.5.post20221221 hydra-core==1.3.2 einops==0.6.0 timm submitit==1.4.5 tensorboard==2.10.0 scipy && \
# pip3 install mmcv-full==1.7.0  mmsegmentation==0.30.0 mmdet==2.25.3 


