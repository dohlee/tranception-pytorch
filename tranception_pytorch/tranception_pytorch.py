import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange

from functools import partial

class SquaredReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x) ** 2

def get_slopes(n, mode="grouped_alibi", verbose=False):
    """
    Function to compute the m constant for each attention head. Code has been adapted from the official ALiBi codebase at:
    https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py
    """
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if mode == "grouped_alibi":
        n = n // 4

    if math.log2(n).is_integer():
        result = get_slopes_power_of_2(n)                   
    else:
        #Workaround when the number of heads is not a power of 2
        closest_power_of_2 = 2**math.floor(math.log2(n))  
        result = get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    if mode == "grouped_alibi":
        result = result * 4
        if verbose:
            print("ALiBi slopes: {}".format(result))

    return result

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
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_heads_per_kernel_size = num_heads // 4
        self.head_dim = embed_dim // num_heads
    
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.to_out = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        self.q_d_conv3 = DepthwiseConvolution(head_dim=self.head_dim, kernel_size=3)
        self.k_d_conv3 = DepthwiseConvolution(head_dim=self.head_dim, kernel_size=3)
        self.v_d_conv3 = DepthwiseConvolution(head_dim=self.head_dim, kernel_size=3)

        self.q_d_conv5 = DepthwiseConvolution(head_dim=self.head_dim, kernel_size=5)
        self.k_d_conv5 = DepthwiseConvolution(head_dim=self.head_dim, kernel_size=5)
        self.v_d_conv5 = DepthwiseConvolution(head_dim=self.head_dim, kernel_size=5)

        self.q_d_conv7 = DepthwiseConvolution(head_dim=self.head_dim, kernel_size=7)
        self.k_d_conv7 = DepthwiseConvolution(head_dim=self.head_dim, kernel_size=7)
        self.v_d_conv7 = DepthwiseConvolution(head_dim=self.head_dim, kernel_size=7)

    def forward(self, x, alibi, causal_mask):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b l (n1 d) -> b n1 l d', n1=self.num_heads), (q, k, v))
        q, k, v = map(lambda t: rearrange(t, 'b (k n2) l d -> b k n2 l d', k=4), (q, k, v))

        q[:, 1] = self.q_d_conv3(q[:, 1])
        k[:, 1] = self.k_d_conv3(q[:, 1])
        v[:, 1] = self.v_d_conv3(q[:, 1])

        q[:, 2] = self.q_d_conv5(q[:, 2])
        k[:, 2] = self.k_d_conv5(q[:, 2])
        v[:, 2] = self.v_d_conv5(q[:, 2])

        q[:, 3] = self.q_d_conv7(q[:, 3])
        k[:, 3] = self.k_d_conv7(q[:, 3])
        v[:, 3] = self.v_d_conv7(q[:, 3])

        q, k, v = map(lambda t: rearrange(t, 'b k n2 l d -> b (k n2) l d', k=4), (q, k, v))

        # Scaled dot product attention + ALiBi position encoding + causal attention masking
        logit = einsum('b n i d, b n j d -> b n i j', q, k) * (self.head_dim ** -0.5) + alibi + causal_mask

        attn = logit.softmax(dim=-1)

        out = einsum('b n i j, b n j d -> b n i d', attn, v)
        out = rearrange(v, 'b n l d -> b l (n d)')

        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, inner_dim=None):
        super().__init__()

        inner_dim = inner_dim or 4 * embed_dim

        self.main = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, inner_dim),
            SquaredReLU(),
            nn.Linear(inner_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
    
    def forward(self, x):
        return self.main(x)
    
class TranceptionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, alibi, causal_mask):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = TranceptionAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.ff = FeedForward(embed_dim)
        self.register_buffer('alibi', alibi)
        self.register_buffer('causal_mask', causal_mask)
    
    def forward(self, x):
        x_res = self.ln1(x)
        x = self.attn(x_res, self.alibi, self.causal_mask) + x_res
        x = self.ln2(x)

        return x

class Tranception(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            num_layers,
            max_length,
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(21, embed_dim)

        # Adopted from https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742
        self.slopes = torch.Tensor(get_slopes(num_heads))
        #In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in the paper). 
        #If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
        #This works because the softmax operation is invariant to translation, and our bias functions are always linear. 
        alibi = self.slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_length).unsqueeze(0).unsqueeze(0).expand(num_heads, -1, -1)
        alibi = alibi.view(num_heads, 1, max_length)
        # NOTE: alibi will be broadcasted to (bsz, num_heads, max_length, max_length) in forward to be added to the logits.
        self.register_buffer('alibi', alibi)
        
        causal_mask = torch.triu(torch.ones(max_length, max_length), diagonal=1) * (-1e-9)
        causal_mask = causal_mask.view(1, 1, max_length, max_length)
        self.register_buffer('causal_mask', causal_mask)

        self.blocks = nn.Sequential(*[
            TranceptionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                alibi=self.alibi,
                causal_mask=self.causal_mask,
            )
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(embed_dim, 21)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.lm_head(x)

        return x
    
    def log_likelihood(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        bsz, length = x.shape[0], x.shape[1]
        out = F.log_softmax(self.forward(x), dim=-1) # (bsz, length, 21)
        mask = torch.zeros((bsz, 21, length), device=x.device).scatter_(1, x.unsqueeze(1), 1.0).permute(0, 2, 1)

        return (out * mask).sum(dim=-1).sum(dim=-1)

if __name__ == '__main__':
    x = torch.randint(0, 21, size=(2, 1024))
    model = Tranception(
        embed_dim=256,
        num_heads=32,
        max_length=1024,
        num_layers=6,
    )

    print(model(x).shape)
    print(model.log_likelihood(x))
