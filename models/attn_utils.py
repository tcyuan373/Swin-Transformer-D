import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention

class PrevDropout(nn.Module):
    def __init__(self, version="fixed", drop_prop=0.5, keep_cnt=197, fix_mask=False):
        super().__init__()
        assert version in ["scale", "fixed"], "version must be one of 'scale', 'fixed'"
        self.version = version
        self.drop_prop = drop_prop
        self.keep_cnt = keep_cnt
        self.fix_mask = fix_mask
        self.mask = None
        self.neg_inf = torch.tensor(-1e12)

    def get_mask(self, x):
        B, N_heads, N1, N2 = x.shape
        if self.version == "scale":
            mask = torch.zeros(B, 1, 1, N2, requires_grad=False).bernoulli(1-self.drop_prop).repeat(1, N_heads, N1, 1).cuda()
            mask.requires_grad = False
        elif self.version == "fixed":
            rows = []
            for _ in range(B):
               row = torch.zeros(1,1,N2, requires_grad=False)
               indices = torch.randperm(N2)[:self.keep_cnt]
               row[0,0,indices] = 1 
               rows.append(row)
            mask = torch.stack(rows, dim=0).repeat(1, N_heads, N1, 1).cuda()
            mask.requires_grad = False
        mask[mask == 0] = self.neg_inf
        mask[mask > 0] = 0
        return mask
    
    def forward(self, x):
        if self.fix_mask and (self.mask is not None):
            return self.mask + x
        mask = self.get_mask(x)
        if self.fix_mask:
            assert self.mask is None
            self.mask = mask
        return mask.to(x.device) + x

class DenseAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, attn_drop=0., proj_drop=0., version="default"):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        if type(qkv_bias) == bool:
            q_bias = k_bias = v_bias = qkv_bias
        else:
            q_bias, k_bias, v_bias = qkv_bias
        
        self.q = nn.Linear(dim, dim, bias=q_bias)
        self.k = nn.Linear(dim, dim, bias=k_bias)
        self.v = nn.Linear(dim, dim, bias=v_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.version = version
    
    def foward(self):
        pass

class Attention_debug(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, proj_drop=proj_drop)
    
    def forward(self, x, debug=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q,k,v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if debug:
            return x, q, k, v
        return x