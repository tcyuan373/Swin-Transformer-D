import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from models.swin_transformer_v2 import *

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
    # our versions

    'dense_deit_RF_pre_tiny_patch16_224', 'dense_deit_RF_post_tiny_patch16_224', 

    'deit_debug_post_tiny_patch16_224',
    'dense_deit_Q_Res_tiny_patch16_224', 
    'dense_deit_V_Res_post_tiny_patch16_224',
    'dense_deit_Q_Res3_uniform_1_relu_tiny_patch16_224',
    'dense_deit_Q_Res3_post_uniform_1_relu_tiny_patch16_224',

    'deit_tiny_norm_exp_none_patch16_224',
    'dense_deit_Q_Res_tiny_patch16_224_none'
]

@register_model
def dense_deit_Q_Res3_uniform_1_relu_tiny_patch16_224(pretrained=False, **kwargs):
    assert not pretrained
    model = SwinTransformerV2_Dense_Res3(
        'Q_Res3', patch_size=16, embed_dim=192, depths=[12], num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_blocks=1, 
        skip_first_emb=True, pos_emb_info={"type": "default"}, 
        attn_init='uniform', attn_init_lo=0.9, attn_init_hi=1.1, attn_act=nn.ReLU, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def keep_swin_config_dense_deit_Q_Res3_uniform_1_relu_tiny_patch16_224(pretrained=False, **kwargs):
    assert not pretrained
    model = SwinTransformerV2_Dense_Res3(
        'Q_Res3', mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_blocks=1, 
        skip_first_emb=True, pos_emb_info={"type": "default"}, 
        attn_init='uniform', attn_init_lo=0.9, attn_init_hi=1.1, attn_act=nn.ReLU, **kwargs)
    model.default_cfg = _cfg()
    return model


model = SwinTransformerV2_Dense_Res3(
        'Q_Res3', mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_blocks=1, 
        skip_first_emb=True, pos_emb_info={"type": "default"}, 
        attn_init='uniform', attn_init_lo=0.9, attn_init_hi=1.1, attn_act=nn.ReLU)

print(model)