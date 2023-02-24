import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange

class PadLeft(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        return F.pad(x, pad=(self.pad, 0))

class DepthwiseConvolution(nn.Module):
    def __init__(self, head_dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.d_conv = nn.Sequential(
            PadLeft(kernel_size - 1),
            nn.Conv1d(head_dim, head_dim, kernel_size, groups=head_dim),
        )
    
    def forward(self, x):
        bsz = x.shape[0]
        x = rearrange(x, 'b n_heads l d -> (b n_heads) d l')
        x = self.d_conv(x)
        x = rearrange(x, '(b n_heads) d l -> b n_heads l d', b=bsz)
        return x
    
class TranceptionAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
        ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
    
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.num_heads), (q, k, v))

        # Scaled dot product attention
        logit = einsum('b i h d, b j h d -> b h i j', q, k) * (self.head_dim ** -0.5)

class Tranception(nn.Module):
    def __init__(
            self,
            embed_dim,
        ):

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(20, embed_dim)

    def forward(self, x):
        x = self.embedding(x)

        pass

if __name__ == '__main__':
    head_dim = 128
    x = torch.randn(2, 8, 1000, head_dim)

    conv = DepthwiseConvolution(head_dim=head_dim, kernel_size=3)
    print(conv(x).shape)

    pass