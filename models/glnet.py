from functools import partial
import math
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, LayerNorm2d, trunc_normal_


_glnet_ckpt_urls= {
    'glnet_4g': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5RzlEalNfQU1xbkpRaVgxP2U9dE9raEhQ/root/content',
    'glnet_9g': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5RUdGUTZrZldWLXdWWmVpP2U9d1d3Ujh6/root/content',
    'glnet_16g': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5amFCeEMzaC1COENIV01tP2U9R2ZSMGtn/root/content',
    'glnet_stl': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5QkZhQUlMRU11X2R0YmJWP2U9OUdoaGkz/root/content',
    'glnet_stl_paramslot': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5RzBlYXQ1ekNFOXVwY1FSP2U9dm1ibW8x/root/content',
    'glnet_4g_tklb': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5QVdjT2RsVFVhMkozdnNYP2U9d0U0Y2gx/root/content',
    'glnet_9g_tklb': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkN5blBFY2ctZkM3WkRDTUV0P2U9Ynd1ZmVS/root/content',
}

def _load_from_url(model:nn.Module, ckpt_key:str, state_dict_key:str='model'):
    url = _glnet_ckpt_urls[ckpt_key]
    checkpoint = torch.hub.load_state_dict_from_url(url=url,
        map_location="cpu", check_hash=True, file_name=f"{ckpt_key}.pth")
    model.load_state_dict(checkpoint[state_dict_key])
    return model

class ResDWConvNCHW(nn.Conv2d):
    def __init__(self, dim, ks:int=3) -> None:
        super().__init__(dim, dim, ks, 1, padding=ks//2, bias=True, groups=dim)

    def forward(self, x:torch.Tensor):
        res = super().forward(x)
        return x + res

class LayerScale(nn.Module):
    def __init__(self, chans, init_value=1e-4, in_format='nlc') -> None:
        super().__init__()
        assert in_format in {'nlc', 'nchw'}
        if in_format == 'nlc':
            self.gamma = nn.Parameter(torch.ones((chans))*init_value, requires_grad=True)
        else: # nchw
            self.gamma = nn.Parameter(torch.ones((1, chans, 1, 1))*init_value, requires_grad=True)

    def forward(self, x:torch.Tensor):
        return self.gamma * x

class MHSA_Block(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, dropout=0.,
                 mlp_ratio:float=4., drop_path:float=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 layerscale=-1) -> None:
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.mha_op = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(mlp_ratio*embed_dim)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio*embed_dim), embed_dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ls1 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nlc') if layerscale > 0 else nn.Identity()
        self.ls2 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nlc') if layerscale > 0 else nn.Identity()

    def forward(self, x:torch.Tensor):
        """
        args:
            x: (bs, len, c) Tensor
        return:
            (bs, len, c) Tensor
        """
        # print(x.dtype)
        shortcut = x
        x = self.norm1(x)
        # print(x.dtype) -> float32 -> RuntimeError: expected scalar type Half but found Float
        # FIXME: below is just a workaround
        # https://github.com/NVIDIA/apex/issues/121#issuecomment-1235109690
        if not self.training:
            x = x.to(shortcut.dtype)

        x, attn_weights = self.mha_op(query=x, key=x, value=x, need_weights=False)
        x = shortcut + self.drop_path(self.ls1(x))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x

class GLMixBlock(nn.Module):
    """
    multihead attention with soft grouping + MLP
    """
    def __init__(self,
        embed_dim:int,
        num_heads:int,
        num_slots:int=64,
        slot_init:str='ada_avgpool',
        slot_scale:float=None,
        scale_mode='learnable',
        local_dw_ks:int=5,
        mlp_ratio:float=4.,
        drop_path:float=0.,
        norm_layer=LayerNorm2d,
        cpe_ks:int=0, # if > 0, the kernel size of the conv pos embedding
        mlp_dw:bool=False,
        layerscale=-1,
        use_slot_attention:bool=True,
        ) -> None:

        super().__init__()
        self.embed_dim = embed_dim
        slot_scale = slot_scale or embed_dim ** (-0.5)
        self.scale_mode = scale_mode
        self.use_slot_attention = use_slot_attention
        assert scale_mode in {'learnable', 'const'}
        if scale_mode  == 'learnable':
            self.slot_scale = nn.Parameter(torch.tensor(slot_scale))
        else: # const
            self.register_buffer('slot_scale', torch.tensor(slot_scale))

        # convolutional position encoding
        self.with_conv_pos_emb = (cpe_ks > 0)
        if self.with_conv_pos_emb:
            self.pos_conv = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=cpe_ks,
                padding=cpe_ks//2, groups=embed_dim)

        # slot initialization
        assert slot_init in {'param', 'ada_avgpool'}
        self.slot_init = slot_init
        if self.slot_init == 'param':
            self.init_slots = nn.Parameter(
                torch.empty(1, num_slots, embed_dim), True)
            torch.nn.init.normal_(self.init_slots, std=.02)
        else:
            self.pool_size = math.isqrt(num_slots)
            # TODO: relax the square number constraint
            assert self.pool_size**2 == num_slots
        
        # spatial mixing
        self.norm1 = norm_layer(embed_dim)
        self.relation_mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True) if use_slot_attention else nn.Identity()
        self.feature_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1), # pseudo qkv linear
            nn.Conv2d(embed_dim, embed_dim, local_dw_ks, padding=local_dw_ks//2, groups=embed_dim), # pseudo attention
            nn.Conv2d(embed_dim, embed_dim, 1), # pseudo out linear
        ) if local_dw_ks > 0 else nn.Identity()
        
        # per-location embedding
        self.norm2 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, int(mlp_ratio*embed_dim), kernel_size=1),
            ResDWConvNCHW(int(mlp_ratio*embed_dim),ks=3) if mlp_dw else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(int(mlp_ratio*embed_dim), embed_dim, kernel_size=1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # layer scale
        self.ls1 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nchw') \
            if layerscale > 0 else nn.Identity()
        self.ls2 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nchw') \
            if layerscale > 0 else nn.Identity()

        # NOTE: hack, for visualization with forward hook
        self.vis_proxy = nn.Identity()
        
    def _forward_relation(self, x:torch.Tensor, init_slots:torch.Tensor):
        """
        x: (bs, c, h, w) tensor
        init_slots: (bs, num_slots, c) tensor
        """
        # FIXME: below is just a workaround
        # https://github.com/NVIDIA/apex/issues/121#issuecomment-1235109690
        # if not self.training:
        #     slot_scale = self.slot_scale.to(x.dtype)

        x_flatten = x.permute(0, 2, 3, 1).flatten(1, 2)
        logits = F.normalize(init_slots, p=2, dim=-1) @ \
            (self.slot_scale*F.normalize(x_flatten, p=2, dim=-1).transpose(-1, -2)) # (bs, num_slots, l)
        # soft grouping
        slots = torch.softmax(logits, dim=-1) @ x_flatten # (bs, num_slots, c)
        # cluster update with mha
        slots, attn_weights = self.relation_mha(query=slots, key=slots, value=slots, need_weights=False)

        if not self.training: # hack, for visualization
            logits, attn_weights = self.vis_proxy((logits, attn_weights))

        # soft ungrouping
        out = torch.softmax(logits.transpose(-1, -2), dim=-1) @ slots # (bs, h*w, c)
    
        out = out.permute(0, 2, 1).reshape_as(x) # (b, c, h, w)
        # # fuse with locally enhanced features
        out = out + self.feature_conv(x)

        return out, slots

    def forward(self, x:torch.Tensor):
        """
        x: (bs, c, h, w) tensor
        init_slots: (bs, num_slots, c) tensor
        """
        if self.slot_init == 'ada_avgpool':
            init_slots = F.adaptive_avg_pool2d(x, output_size=self.pool_size
                ).permute(0, 2, 3, 1).flatten(1, 2)
        else:
            init_slots = self.init_slots

        # print(x.dtype, init_slots.dtype) -> float16 float32
        if self.with_conv_pos_emb:
            x = x + self.pos_conv(x)

        shortcut = x
        x, updt_slots = self._forward_relation(self.norm1(x), init_slots)
        x = shortcut + self.drop_path(self.ls1(x))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        return x
    
    def extra_repr(self) -> str:
        return f"scale_mode={self.scale_mode}, "\
               f"slot_init={self.slot_init}, "\
               f"use_slot_attention={self.use_slot_attention}"


class MHSA_NCHW_Block(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, dropout=0.,
                 mlp_ratio:float=4., drop_path:float=0., norm_layer=LayerNorm2d,
                 mlp_dw:bool=False, cpe_ks:int=0, # if > 0, the kernel size of the conv pos embedding
                 layerscale=-1,
                ) -> None:
        super().__init__()
        # convolutional position encoding
        self.with_conv_pos_emb = (cpe_ks > 0)
        if self.with_conv_pos_emb:
            self.pos_conv = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=cpe_ks,
                padding=cpe_ks//2, groups=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mha_op = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, int(mlp_ratio*embed_dim), kernel_size=1),
            ResDWConvNCHW(int(mlp_ratio*embed_dim),ks=3) if mlp_dw else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(int(mlp_ratio*embed_dim), embed_dim, kernel_size=1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ls1 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nlc') if layerscale > 0 else nn.Identity()
        self.ls2 = LayerScale(chans=embed_dim, init_value=layerscale, in_format='nchw') if layerscale > 0 else nn.Identity()

    def forward(self, x:torch.Tensor):
        """
        args:
            x: (bs, c, h, w) Tensor
        return:
            (bs, c, h, w) Tensor
        """
        # Conv. Pos. Embedding
        if self.with_conv_pos_emb:
            x = x + self.pos_conv(x)

        # reshape tp nlc format
        nchw_shape = x.size()
        x = x.permute(0, 2, 3, 1).flatten(1, 2) # (bs, h*w, c)

        # forward attention block
        # print(x.dtype)
        shortcut = x
        x = self.norm1(x)
        # print(x.dtype) -> float32 -> RuntimeError: expected scalar type Half but found Float
        # FIXME: below is just a workaround
        # https://github.com/NVIDIA/apex/issues/121#issuecomment-1235109690
        if not self.training:
            x = x.to(shortcut.dtype)
        x, attn_weights = self.mha_op(query=x, key=x, value=x, need_weights=False)
        x = shortcut + self.drop_path(self.ls1(x))
        
        # forward mlp block
        x = x.permute(0, 2, 1).reshape(nchw_shape)
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        return x

class BasicLayer(nn.Module):
    """
    Stack several Blocks (a stage of transformer)
    """
    def __init__(self, 
        dim, num_heads, depth:int,
        mlp_ratio=4., drop_path=0.,
        ################
        mixing_mode='glmix', # {'mha',  'sgmha', 'dw', 'glmix'}
        local_dw_ks=5, # kernel size of dw conv, for 'dw' and 'glmix'
        slot_init:str='ada_avgpool', # {'param', 'conv', 'pool', 'ada_avgpool'}
        num_slots:int=64, # to control number of slots
        use_slot_attention:bool=True,
        cpe_ks:int=0,
        mlp_dw:bool=False,
        ##############
        layerscale=-1
        ):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mixing_mode = mixing_mode

        # instantiate blocks
        if self.mixing_mode == 'mha':
            self.blocks = nn.ModuleList([
                MHSA_Block(
                    embed_dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layerscale=layerscale
                ) for i in range(depth)])
        elif self.mixing_mode == 'mha_nchw':
            self.blocks = nn.ModuleList([
                MHSA_NCHW_Block(
                    embed_dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    cpe_ks=cpe_ks,
                    mlp_dw=mlp_dw,
                    layerscale=layerscale
                ) for i in range(depth)])
        elif self.mixing_mode == 'glmix':
            self.blocks = nn.ModuleList([
                GLMixBlock(
                    embed_dim=dim,
                    num_heads=num_heads,
                    num_slots=num_slots,
                    slot_init=slot_init,
                    local_dw_ks=local_dw_ks,
                    use_slot_attention=use_slot_attention,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    cpe_ks=cpe_ks,
                    mlp_dw=mlp_dw,
                    layerscale=layerscale
                )for i in range(depth)])
        elif self.mixing_mode == 'glmix.mha_nchw': # hybrid
            self.blocks = nn.ModuleList([
                GLMixBlock(
                    embed_dim=dim,
                    num_heads=num_heads,
                    num_slots=num_slots,
                    slot_init=slot_init,
                    local_dw_ks=local_dw_ks,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    cpe_ks=cpe_ks,
                    mlp_dw=mlp_dw,
                    layerscale=layerscale
                ) if i % 2 == 0 else \
                MHSA_NCHW_Block(
                    embed_dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    cpe_ks=cpe_ks,
                    mlp_dw=mlp_dw,
                    layerscale=layerscale
                ) for i in range(depth)])
        else:
            raise ValueError('unknown block type')

    def forward(self, x:torch.Tensor):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # print(x.dtype)
        if self.mixing_mode == 'mha':
            nchw_shape = x.size()
            x = x.permute(0, 2, 3, 1).flatten(1, 2)
            for blk in self.blocks:
                x = blk(x) # (bs, len, c)
            x = x.transpose(1, 2).reshape(nchw_shape)
        else: # the input output are both nchw format
            for blk in self.blocks:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"

# In SwinTransformer, stem is in the order of proj -> norm
# https://github.com/microsoft/Swin-Transformer/blob/2cb103f2de145ff43bb9f6fc2ae8800c24ad04c6/models/swin_transformer.py#L437

# while, middle patch merging is in the order norm -> proj
# https://github.com/microsoft/Swin-Transformer/blob/2cb103f2de145ff43bb9f6fc2ae8800c24ad04c6/models/swin_transformer.py#L315

class NonOverlappedPatchEmbeddings(nn.ModuleList):
    def __init__(self, embed_dims:Iterable[int], in_chans=3,
                       midd_order='norm.proj',
                       norm_layer=nn.BatchNorm2d) -> None:
        assert midd_order in {'norm.proj', 'proj.norm'}
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0],kernel_size=(4, 4), stride=(4, 4)),
            norm_layer(embed_dims[0])
        )
        modules = [stem]
        for i in range(3):
            if midd_order == 'norm.proj':
                transition = nn.Sequential(
                    norm_layer(embed_dims[i]), 
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=(2, 2), stride=(2, 2)),
                )
            else:
                transition = nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=(2, 2), stride=(2, 2)),
                    norm_layer(embed_dims[i+1])
                )
            modules.append(transition)
        super().__init__(modules)


# https://github.com/rayleizhu/BiFormer/blob/1697bbbeafb8680524898f1dcaac10defd0604be/models/biformer.py#L186
# https://github.com/rayleizhu/BiFormer/blob/1697bbbeafb8680524898f1dcaac10defd0604be/models/biformer.py#L200
class OverlappedPacthEmbeddings(nn.ModuleList):
    def __init__(self, embed_dims:Iterable[int],
        in_chans=3,
        deep_stem=True,
        dual_patch_norm=False,
        midd_order='proj.norm',
        norm_layer=nn.BatchNorm2d
        ) -> None:
        assert midd_order in {'norm.proj', 'proj.norm'}
        if deep_stem:
            stem = nn.Sequential(
                LayerNorm2d(in_chans) if dual_patch_norm else nn.Identity(),
                nn.Conv2d(in_chans, embed_dims[0] // 2, kernel_size=3, stride=2, padding=1),
                norm_layer(embed_dims[0] // 2),
                nn.GELU(),
                nn.Conv2d(embed_dims[0] // 2, embed_dims[0], kernel_size=3, stride=2, padding=1),
                norm_layer(embed_dims[0])
            )
        else:
            stem = nn.Sequential(
                LayerNorm2d(in_chans) if dual_patch_norm else nn.Identity(),
                nn.Conv2d(in_chans, embed_dims[0] // 2, kernel_size=7, stride=4, padding=3),
                norm_layer(embed_dims[0] // 2),
            )
        modules = [stem]
        for i in range(3):
            if midd_order == 'norm.proj':
                transition = nn.Sequential(
                    norm_layer(embed_dims[i]), 
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=3, stride=2, padding=1),
                )
            else:
                transition = nn.Sequential(
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=3, stride=2, padding=1),
                    norm_layer(embed_dims[i+1])
                )
            modules.append(transition)
        super().__init__(modules)


class GLNet(nn.Module):
    """
    vision transformer with soft grouping
    """
    def __init__(self,
        in_chans=3,
        num_classes=1000,
        depth=[2, 2, 6, 2],
        embed_dim=[96, 192, 384, 768],
        head_dim=32, qk_scale=None,
        drop_path_rate=0., drop_rate=0.,
        use_checkpoint_stages=[],
        mlp_ratios=[4, 4, 4, 4],
        norm_layer=LayerNorm2d,
        pre_head_norm_layer=None,
        ######## glnet specific ############
        mixing_modes=('glmix', 'glmix', 'glmix', 'mha'), # {'mha', 'glmix',  'glmix.mha_nchw', 'mha_nchw'}
        local_dw_ks=5, # kernel size of dw conv
        slot_init:str='param', #{'param', 'conv', 'pool', 'ada_pool'}
        num_slots:int=64, # to control number of slots
        cpe_ks:int=0,
        #######################################
        downsample_style:str='non_ovlp', # {'non_ovlp', 'ovlp'}
        transition_layout:str='proj.norm', # {'norm.proj', 'proj.norm'}
        dual_patch_norm:bool=False,
        mlp_dw:bool=False,
        layerscale:float=-1.,
        ###################
        **unused_kwargs
        ):
        super().__init__()
        print(f"unused_kwargs in model initilization: {unused_kwargs}.")

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        assert downsample_style in {'non_ovlp', 'ovlp'}
        if downsample_style=='ovlp':
            self.downsample_layers = OverlappedPacthEmbeddings(
                embed_dims=embed_dim, in_chans=in_chans, norm_layer=norm_layer,
                midd_order=transition_layout,
                dual_patch_norm=dual_patch_norm)
        else:
            self.downsample_layers = NonOverlappedPatchEmbeddings(
                embed_dims=embed_dim, in_chans=in_chans, norm_layer=norm_layer,
                midd_order=transition_layout
            )
        ##########################################################################
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads= [dim // head_dim for dim in embed_dim]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        local_dw_ks = [local_dw_ks,]*4 if isinstance(local_dw_ks, int) else local_dw_ks

        for i in range(4):
            stage = BasicLayer(
                dim=embed_dim[i],
                depth=depth[i],
                num_heads=nheads[i], 
                mlp_ratio=mlp_ratios[i],
                drop_path=dp_rates[sum(depth[:i]):sum(depth[:i+1])],
                ####### glnet specific ########
                mixing_mode=mixing_modes[i],
                local_dw_ks=local_dw_ks[i],
                slot_init=slot_init,
                num_slots=num_slots,
                cpe_ks=cpe_ks,
                mlp_dw=mlp_dw,
                layerscale=layerscale
                ##################################
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)

        ##########################################################################
        pre_head_norm = pre_head_norm_layer or norm_layer 
        self.norm = pre_head_norm(embed_dim[-1])
        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x:torch.Tensor):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        return x

    def forward(self, x:torch.Tensor):
        x = self.forward_features(x) # (n, c, h, w)
        # x = x.flatten(2).mean(-1)
        x = x.mean([2, 3])
        x = self.head(x)
        return x

@register_model
def glnet_stl(pretrained=False, pretrained_cfg=None,
        pretrained_cfg_overlay=None, **kwargs):
    model = GLNet(
        depth=[2, 2, 6, 2],
        embed_dim=[96, 192, 384, 768],
        mlp_ratios=[4, 4, 4, 4],
        head_dim=32,
        norm_layer=nn.BatchNorm2d,
        ######## glnet specific ############
        mixing_modes=('glmix', 'glmix', 'glmix', 'mha'), # {'mha', 'glmix',  'glmix.mha_nchw', 'mha_nchw'}
        local_dw_ks=5, # kernel size of dw conv
        slot_init='ada_avgpool', #{'param', 'ada_avgpool'}
        num_slots=64, # to control number of slots
        transition_layout='norm.proj',
        #######################################
        **kwargs)
    if pretrained:
        model = _load_from_url(model, ckpt_key='glnet_stl')
    return model


@register_model
def glnet_stl_paramslot(pretrained=False, pretrained_cfg=None,
        pretrained_cfg_overlay=None, **kwargs):
    model = GLNet(
        depth=[2, 2, 6, 2],
        embed_dim=[96, 192, 384, 768],
        mlp_ratios=[4, 4, 4, 4],
        head_dim=32,
        norm_layer=nn.BatchNorm2d,
        ######## glnet specific ############
        mixing_modes=('glmix', 'glmix', 'glmix', 'mha'), # {'mha', 'glmix',  'glmix.mha_nchw', 'mha_nchw'}
        local_dw_ks=5, # kernel size of dw conv
        slot_init='param', #{'param', 'ada_avgpool'}
        num_slots=64, # to control number of slots
        transition_layout='norm.proj',
        #######################################
        **kwargs)
    if pretrained:
        model = _load_from_url(model, ckpt_key='glnet_stl_paramslot')
    return model


@register_model
def glnet_4g(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = GLNet(
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
        model = _load_from_url(model, ckpt_key='glnet_4g')
    return model


@register_model
def glnet_9g(pretrained=False, pretrained_cfg=None,
            pretrained_cfg_overlay=None, **kwargs):
    model = GLNet(
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
        model = _load_from_url(model, ckpt_key='glnet_9g')
    return model


@register_model
def glnet_16g(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = GLNet(
        depth=[4, 4, 18, 4],
        embed_dim=[128, 256, 512, 1024],
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
        layerscale=1e-4,
        #######################################
        **kwargs)
    if pretrained:
        # NOTE: only for glnet_16g model, the ema checkpoint has slightly better
        # peformance (85.0) than that of regular checkpoint (84.9, not from the same epoch)
        model = _load_from_url(model, ckpt_key='glnet_16g', state_dict_key='model_ema')
    return model
