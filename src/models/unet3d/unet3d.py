import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from typing import Optional

# --- Helper functions ---
def exists(x):
    """Check if a value is not None."""
    return x is not None

def default(val, d):
    """Return 'val' if it exists, otherwise return 'd'."""
    if exists(val):
        return val
    return d() if callable(d) else d

# --- 3D Module Definitions ---

class WeightStandardizedConv3d(nn.Conv3d):
    """
    Applies weight standardization to a 3D convolutional layer.
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        
        mean = torch.mean(weight, dim=[1, 2, 3, 4], keepdim=True)
        var = torch.var(weight, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv3d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

def Upsample3d(dim, dim_out=None, scale_factor=2):
    """Upsampling block for 3D data."""
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode="nearest"),
        nn.Conv3d(dim, default(dim_out, dim), kernel_size=3, padding=1),
    )

def Downsample3d(dim, dim_out=None):
    """Downsampling block for 3D data using a strided convolution."""
    return nn.Conv3d(dim, default(dim_out, dim), kernel_size=4, stride=2, padding=1)

class LayerNorm3d(nn.Module):
    """Layer Normalization adapted for 5D tensors (B, C, D, H, W)."""
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

# --- Attention (Assuming flattened spatial dimensions) ---
# The original attention modules are adapted to flatten the 3D spatial dimensions
# before computing attention, a common approach for transformers in vision.

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        shape = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c') # Flatten
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return rearrange(self.to_out(out), 'b (d h w) c -> b c d h w', **{k: v for k, v in zip('dhw', shape[2:])})


class LinearAttention(Attention):
    """A memory-efficient version of attention."""
    def forward(self, x):
        shape = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c') # Flatten
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return rearrange(self.to_out(out), 'b (d h w) c -> b c d h w', **{k: v for k, v in zip('dhw', shape[2:])})

# --- Generic helper modules ---

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_class):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# Simplified time embedder for demonstration
def get_time_embedder(time_dim, pos_emb_dim, *args, **kwargs):
    return nn.Sequential(nn.Linear(pos_emb_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

# --- 3D Building Blocks ---

class Block3d(nn.Module):
    def __init__(self, dim, dim_out, groups=8, dropout=0.0):
        super().__init__()
        self.proj = WeightStandardizedConv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock3d(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, **kwargs):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim) else None
        )
        self.block1 = Block3d(dim, dim_out, groups=groups)
        self.block2 = Block3d(dim_out, dim_out, groups=groups)
        self.residual_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            # Reshape for 3D: (b, c) -> (b, c, 1, 1, 1)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.residual_conv(x)


# --- 3D U-Net Model ---

class Unet3d(nn.Module):
    def __init__(self, dim, num_input_channels, num_output_channels, dim_mults=(1, 2, 4, 8), **kwargs):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        
        init_dim = default(kwargs.get('init_dim'), dim)
        self.init_conv = nn.Conv3d(num_input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock3d, **kwargs)
        
        # Encoder
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in), LayerNorm3d)),
                Downsample3d(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding=1)
            ]))
        
        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim), LayerNorm3d))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        # Decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out), LayerNorm3d)),
                Upsample3d(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Conv3d(dim, self.num_output_channels, 1)

    def forward(self, x, time=None, **kwargs):
        x = self.init_conv(x)
        r = x.clone()

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb=time)
            h.append(x)
            x = block2(x, time_emb=time)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb=time)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb=time)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, time_emb=time)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, time_emb=time)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, time_emb=time)
        return self.final_conv(x)


# --- Example Usage ---
if __name__ == "__main__":
    model = Unet3d(
        dim=64,
        num_input_channels=1,
        num_output_channels=1,
        dim_mults=(1, 2, 4),
        groups=8 # For GroupNorm in ResnetBlock3d
    )

    # Dummy 3D input: (B, C, D, H, W)
    dummy_input = torch.rand(1, 1, 16, 64, 64)
    print(f"Input shape: {dummy_input.shape}")

    # Get model output
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    assert output.shape == dummy_input.shape
    print("\nModel instantiated and tested successfully.")
