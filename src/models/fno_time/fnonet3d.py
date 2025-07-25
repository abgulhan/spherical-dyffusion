# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# 3D Fourier Neural Operator Net (drop-in replacement for SphericalFourierNeuralOperatorNet)
#
# This model uses the Modulus FNO core for 3D data, but follows the modular/block structure of sfnonet.py.

import torch
import torch.nn as nn
from functools import partial
from typing import Any, Optional
from torch import amp
from einops import rearrange
from modulus.models.fno.fno import FNO3DEncoder
from src.models._base_model import BaseModel
from src.models.modules.misc import get_time_embedder



class FNO3DBlock(nn.Module):
    """
    Custom FNO3D block with time embedding support, similar to SFNO blocks.
    """
    def __init__(
        self,
        embed_dim: int,
        num_fno_modes: int = 16,
        activation_fn=nn.GELU(),
        normalization_layer: str = "instance_norm",
        dropout: float = 0.0,
        time_emb_dim: int = None,
        time_scale_shift_before_filter: bool = True,
        use_mlp: bool = False,
        mlp_ratio: float = 2.0,
        num_layers: int = 12,
        padding: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_scale_shift_before_filter = time_scale_shift_before_filter
        
        # Normalization layers
        if normalization_layer == "instance_norm":
            self.norm0 = nn.InstanceNorm3d(embed_dim)
            self.norm1 = nn.InstanceNorm3d(embed_dim)
        elif normalization_layer == "layer_norm":
            # Note: Layer norm for 3D requires spatial dimensions to be known
            # For now, we'll use instance norm as a fallback
            self.norm0 = nn.InstanceNorm3d(embed_dim)
            self.norm1 = nn.InstanceNorm3d(embed_dim)
        elif normalization_layer == "none":
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization layer {normalization_layer}")
        
        # Time embedding MLP
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, embed_dim * 2),  # 2 for scale and shift
                activation_fn,
            )
        else:
            self.time_mlp = None
        
        # FNO spectral convolution core
        # self.spectral_conv = SpectralConv3d(
        #     in_channels=embed_dim,
        #     out_channels=embed_dim,
        #     modes1=num_fno_modes,
        #     modes2=num_fno_modes,
        #     modes3=num_fno_modes,
        # )
        self.fno_encoder = FNO3DEncoder(
            in_channels=embed_dim,
            num_fno_layers=1,  # Single layer for this block, use multiple blocks for multiple layers
            fno_layer_size=embed_dim,
            num_fno_modes=num_fno_modes,
            padding=padding, # TODO calculate padding based on inputs
            activation_fn=nn.GELU(),
            coord_features=True,
        )
        
        # Optional MLP
        if use_mlp:
            mlp_hidden = int(embed_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Conv3d(embed_dim, mlp_hidden, 1),
                activation_fn,
                nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
                nn.Conv3d(mlp_hidden, embed_dim, 1),
            )
        else:
            self.mlp = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
    
    def time_scale_shift(self, x, time_emb):
        """Apply time-based scale and shift transformation."""
        #print(f"==> time_emb_input {time_emb}")
        assert time_emb is not None, "time_emb is None but time_scale_shift is called"
        time_emb = self.time_mlp(time_emb)
        time_emb = rearrange(time_emb, "b c -> b c 1 1 1")  # Add spatial dimensions for 3D
        scale, shift = time_emb.chunk(2, dim=1)  # split into scale and shift (channel dim)
        x = x * (scale + 1) + shift
        # print(f"Applied time scale/shift: scale shape {scale.shape}, shift shape {shift.shape}")  # Debug log
        # print(f"===> time_emb: {list(time_emb.squeeze().tolist())}")  # Debug log
        # print(f"===> scale: {list(scale.squeeze().tolist())}")  # Debug log
        # print(f"===> shift: {list(shift.squeeze().tolist())}")  # Debug log
        return x
    
    def forward(self, x, time_emb=None):
        # First normalization
        x_norm = self.norm0(x)
        
        # Apply time conditioning before filter if specified
        if self.time_scale_shift_before_filter and self.time_mlp is not None and time_emb is not None:
            x_norm = self.time_scale_shift(x_norm, time_emb)
        
        # Spectral convolution (simplified FNO operation)
        with amp.autocast(device_type='cuda', enabled=False):  # Disable autocast for FNO core
            x_filtered = self.fno_encoder(x_norm)
        
        # Residual connection
        x = x + x_filtered
        
        # Second normalization
        x_norm = self.norm1(x)
        
        # Apply time conditioning after filter if specified
        if not self.time_scale_shift_before_filter and self.time_mlp is not None and time_emb is not None:
            x_norm = self.time_scale_shift(x_norm, time_emb)
        
        # Optional MLP
        x_mlp = self.mlp(x_norm)
        
        # Residual connection and dropout
        x = x + self.dropout(x_mlp)
        
        return x
from src.models.modules.misc import get_time_embedder

class FourierNeuralOperatorNet3D(BaseModel):
    """
    3D Fourier Neural Operator Network (Cartesian)
    Drop-in replacement for SphericalFourierNeuralOperatorNet, but for 3D data.
    Now includes time embedding support similar to SFNO.
    """
    def __init__(
        self,
        in_chans: int = None,
        out_chans: int = None,
        img_shape: tuple = None,
        num_input_channels: int = None,  # Alternative name for in_chans
        num_output_channels: int = None,  # Alternative name for out_chans
        spatial_shape_in: tuple = None,  # Alternative name for img_shape
        spatial_shape_out: tuple = None,  # Alternative name for img_shape
        embed_dim: int = 64,
        num_layers: int = 4,
        num_fno_modes: int = 16,
        padding: int = 8,
        padding_type: str = "constant",
        activation_function: str = "gelu",
        normalization_layer: str = "instance_norm",
        dropout: float = 0.0,
        big_skip: bool = True,
        use_mlp: bool = False,
        mlp_ratio: float = 2.0,
        pos_embed: bool = False,
        # Time embedding parameters (similar to SFNO)
        with_time_emb: bool = False,
        time_dim_mult: int = 2,
        time_rescale: bool = False,
        time_scale_shift_before_filter: bool = True,
        loss_function: str = "mean_squared_error",
        loss_function_weights: Optional[dict] = None,
        datamodule_config: Optional[Any] = None,
        debug_mode: bool = False,
        name: str = "",
        verbose: bool = True,
        **kwargs,
    ):
        # Handle both naming conventions
        in_chans = in_chans if in_chans is not None else num_input_channels
        out_chans = out_chans if out_chans is not None else num_output_channels
        img_shape = img_shape if img_shape is not None else (spatial_shape_in or spatial_shape_out)
        
        if in_chans is None or out_chans is None or img_shape is None:
            raise ValueError(f"Must provide in_chans/num_input_channels, out_chans/num_output_channels, and img_shape/spatial_shape_in. "
                           f"Got: in_chans={in_chans}, out_chans={out_chans}, img_shape={img_shape}")
        
        super().__init__(
            num_input_channels=in_chans,
            num_output_channels=out_chans,
            spatial_shape_in=img_shape,
            spatial_shape_out=img_shape,
            loss_function=loss_function,
            loss_function_weights=loss_function_weights,
            datamodule_config=datamodule_config,
            debug_mode=debug_mode,
            name=name,
            verbose=verbose,
        )
        self.img_shape = img_shape
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.big_skip = big_skip
        self.use_mlp = use_mlp
        self.pos_embed = pos_embed
        self.with_time_emb = with_time_emb
        self.time_scale_shift_before_filter = time_scale_shift_before_filter

        # Activation
        if activation_function == "relu":
            self.activation = nn.ReLU()
        elif activation_function == "gelu":
            self.activation = nn.GELU()
        elif activation_function == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # Time embedding (similar to SFNO implementation)
        self.time_dim = None
        if with_time_emb:
            self.log_text.info("Using time embedding with FNO3D")
            pos_emb_dim = self.embed_dim
            sinusoidal_embedding = "true"
            self.time_dim = self.embed_dim * time_dim_mult
            self.time_rescale = time_rescale
            self.min_time, self.max_time = None, None
            self.time_scaler = 1.0
            self.time_shift = 0.0
            self.time_emb_mlp = get_time_embedder(self.time_dim, pos_emb_dim, sinusoidal_embedding)
        else:
            self.time_rescale = False

        # Normalization
        if normalization_layer == "instance_norm":
            self.norm = nn.InstanceNorm3d(embed_dim)
        elif normalization_layer == "layer_norm":
            self.norm = nn.LayerNorm([embed_dim, *img_shape])
        elif normalization_layer == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization layer {normalization_layer}")

        # Dropout
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        # Encoder: project input to embed_dim
        self.encoder = nn.Conv3d(in_chans, embed_dim, kernel_size=1)

        # Create custom FNO blocks with time conditioning instead of using Modulus FNO directly
        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            block = FNO3DBlock(
                embed_dim=embed_dim,
                num_fno_modes=num_fno_modes,
                activation_fn=self.activation,
                normalization_layer=normalization_layer,
                dropout=dropout,
                time_emb_dim=self.time_dim if with_time_emb else None,
                time_scale_shift_before_filter=time_scale_shift_before_filter,
                use_mlp=use_mlp,
                mlp_ratio=mlp_ratio,
                padding=padding,
            )
            self.blocks.append(block)

        # Optional MLP head (separate from block MLPs)
        if use_mlp:
            mlp_hidden = int(embed_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Conv3d(embed_dim, mlp_hidden, 1),
                self.activation,
                nn.Conv3d(mlp_hidden, embed_dim, 1),
            )
        else:
            self.mlp = nn.Identity()

        # Decoder: project embed_dim to out_chans
        self.decoder = nn.Conv3d(embed_dim + (in_chans if big_skip else 0), out_chans, kernel_size=1)

        # Optional positional embedding
        if pos_embed:
            self.pos_embed_param = nn.Parameter(torch.zeros(1, embed_dim, *img_shape))
            nn.init.trunc_normal_(self.pos_embed_param, std=0.02)
        else:
            self.pos_embed_param = None

    def set_min_max_time(self, min_time: float, max_time: float):
        """Use time stats to rescale time input to [0, 1000] (similar to SFNO)."""
        self.min_time, self.max_time = min_time, max_time
        if self.time_rescale:
            self.time_scaler = 1000.0 / (max_time - min_time)
            self.time_shift = -min_time
            # if hasattr(self, 'log_text'):
            self.log_text.info(
                f"Time rescaling: min_time: {min_time}, max_time: {max_time}, "
                f"time_scaler: {self.time_scaler}, time_shift: {self.time_shift}"
            )
        else:
            # if hasattr(self, 'log_text'):
            self.log_text.info(f"Time stats will be checked: min_time: {min_time}, max_time: {max_time}")

    def forward_features(self, x, time=None):
        """Forward pass through FNO blocks with optional time conditioning."""
        if self.with_time_emb:
            assert (
                self.min_time is not None and self.max_time is not None
            ), "min_time and max_time must be set before using time embedding"
            assert (self.min_time <= time).all() and (
                time <= self.max_time
            ).all(), f"time must be in [{self.min_time}, {self.max_time}], but time is {time}"
            if self.time_rescale:
                #time = time * self.time_scaler + self.time_shift
                time = (time + self.time_shift) * self.time_scaler
            t_repr = self.time_emb_mlp(time)
        else:
            t_repr = None

        for i, blk in enumerate(self.blocks):
            x = blk(x, time_emb=t_repr)
        return x, t_repr

    def forward(self, x, time=None, condition=None, return_time_emb: bool = False, **kwargs):
        # Only extract 'dynamics' if x is a dict (for compatibility)
        if isinstance(x, dict):
            if 'dynamics' in x:
                extra_keys = set(x.keys()) - {'dynamics'}
                if extra_keys:
                    raise ValueError(f"Input dict to FNO3D model contains unsupported keys besides 'dynamics': {extra_keys}")
                x = x['dynamics']
            else:
                raise ValueError("Input dict to FNO3D model must contain 'dynamics' key.")
        # If x is a tuple or list, extract the first element
        if isinstance(x, (list, tuple)):
            x = x[0]
        
        # Save original dtype for mixed precision compatibility
        dtype = x.dtype
        
        # Convert to float32 for processing (similar to SFNO)
        x = x.float()
        
        # Save big skip connection
        residual = x if self.big_skip else None
        
        # Encoder: project input to embed_dim
        x = self.encoder(x)
        
        # Add positional embedding if available
        if self.pos_embed_param is not None:
            x = x + self.pos_embed_param
        
        # Initial normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        # Forward through time-aware FNO blocks
        x, t_repr = self.forward_features(x, time)
        
        # Apply MLP if enabled
        x = self.mlp(x)
        
        # Apply big skip connection
        if self.big_skip and residual is not None:
            x = torch.cat([x, residual], dim=1)
        
        # Decoder: project back to output channels
        x = self.decoder(x)
        
        # Cast back to original dtype for mixed precision compatibility
        x = x.to(dtype)
        
        if return_time_emb:
            return x, t_repr
        else:
            return x
