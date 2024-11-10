import torch
from torch import nn
import numpy as np

from timm.models import register_model

from .glnet import GLNet
from .glnet import _load_from_url

def rand_bbox(size, lam, scale=1):
    # size: NCHW
    # NOTE: the two local variables are wrongly named,
    # but we need to follow the wrong protocol to using token labeling
    W = size[2] // scale
    H = size[3] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class GLNet_tklb(GLNet):
    def __init__(self, **glnet_kwargs):
        super().__init__(**glnet_kwargs)
        # self.tklb_return_dense = True
        # self.tklb_mix_token = True
        self.tklb_beta = 1.0
        self.tklb_pooling_scale = 8 # (1/4) / (1/32) = 8
        self.tklb_aux_head = nn.Linear(self.embed_dim[-1], self.num_classes)
    
    def forward(self, x:torch.Tensor):
        x = self.downsample_layers[0](x) # NCHW
        
        # cutmix
        if self.training: 
            lam = np.random.beta(self.tklb_beta, self.tklb_beta)
            patch_h, patch_w = x.size(2)//self.tklb_pooling_scale, \
                x.size(3)//self.tklb_pooling_scale # corresponding feature map size in final stage
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(),
                lam=lam, scale=self.tklb_pooling_scale) # corresponding crop region in final stage 
            sbbx1,sbby1,sbbx2,sbby2 = self.tklb_pooling_scale*bbx1, self.tklb_pooling_scale*bby1,\
                self.tklb_pooling_scale*bbx2, self.tklb_pooling_scale*bby2 # crop region in patch embedding
            ## intra batch cutmix
            temp_x = x.clone()
            temp_x[:, :, sbbx1:sbbx2, sbby1:sbby2] = x.flip(0)[:, :, sbbx1:sbbx2, sbby1:sbby2]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0
        
        # normal forward
        for i in range(4):
            if i != 0: # stage 0 downsampling has been done
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        x = x.flatten(2).transpose(-1, -2) # (bs, h*w, c)
        cls_tokens = x.mean(dim=1, keepdim=False) # (bs, c)
        x_cls = self.head(cls_tokens) # (bs, num_classes), normal head
        x_aux:torch.Tensor = self.tklb_aux_head(x) # (bs, h*w, num_classes), token_labeling head

        # inverse cutmix
        if self.training:
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
            temp_x = x_aux.clone()
            temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
            x_aux = temp_x
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])
            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)
        else:
            return x_cls + 0.5 * x_aux.max(1)[0]


@register_model
def glnet_4g_tklb(pretrained=False, pretrained_cfg=None,
                pretrained_cfg_overlay=None, **kwargs):
    model = GLNet_tklb(
        depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512],
        mlp_ratios=[3, 3, 3, 3],
        head_dim=32,
        norm_layer=nn.BatchNorm2d,
        ######## glnet specific ############
        mixing_modes=('glmix', 'glmix', 'glmix.mha_nchw', 'mha_nchw'),
        local_dw_ks=5, # kernel size of dw conv
        slot_init='ada_avgpool', #{'param', 'conv', 'pool', 'ada_pool'}
        num_slots=64, # to control number of slots
        #######################################
        cpe_ks=3,
        downsample_style='ovlp', # overlapped patch embedding
        transition_layout='proj.norm',
        mlp_dw=True,
        #######################################
        **kwargs)
    if pretrained:
        model = _load_from_url(model, ckpt_key='glnet_4g_tklb')
    return model

@register_model
def glnet_9g_tklb(pretrained=False, pretrained_cfg=None,
                      pretrained_cfg_overlay=None, **kwargs):
    model = GLNet_tklb(
        depth=[4, 4, 18, 4],
        embed_dim=[96, 192, 384, 768],
        mlp_ratios=[3, 3, 3, 3],
        head_dim=32,
        norm_layer=nn.BatchNorm2d,
        ######## glnet specific ############
        mixing_modes=('glmix', 'glmix', 'glmix.mha_nchw', 'mha_nchw'),
        local_dw_ks=5, # kernel size of dw conv
        slot_init='ada_avgpool', #{'param', 'conv', 'pool', 'ada_pool'}
        num_slots=64, # to control number of slots
        #######################################
        cpe_ks=3,
        downsample_style='ovlp', # overlapped patch embedding
        transition_layout='proj.norm',
        mlp_dw=True,
        #######################################
        **kwargs)
    if pretrained:
        model = _load_from_url(model, ckpt_key='glnet_9g_tklb')
    return model

# @register_model
# def glnet_16g_tklb(pretrained=False, pretrained_cfg=None,
#                       pretrained_cfg_overlay=None, **kwargs):
#     model = GLNet_tklb(
#         depth=[4, 4, 18, 4],
#         embed_dim=[128, 256, 512, 1024],
#         mlp_ratios=[3, 3, 3, 3],
#         head_dim=32,
#         norm_layer=nn.BatchNorm2d,
#         ######## glnet specific ############
#         mixing_modes=('glmix', 'glmix', 'glmix.mha_nchw', 'mha_nchw'),
#         local_dw_ks=5, # kernel size of dw conv
#         slot_init='ada_avgpool', #{'param', 'conv', 'pool', 'ada_pool'}
#         num_slots=64, # to control number of slots
#         #######################################
#         cpe_ks=3,
#         downsample_style='ovlp', # overlapped patch embedding
#         transition_layout='proj.norm',
#         mlp_dw=True,
#         layerscale=1e-4,
#         #######################################
#         **kwargs)
#     return model
