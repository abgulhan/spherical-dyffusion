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
from modulus.models.fno.fno import FNO3DEncoder
from src.models._base_model import BaseModel

class FourierNeuralOperatorNet3D(BaseModel):
    """
    3D Fourier Neural Operator Network (Cartesian)
    Drop-in replacement for SphericalFourierNeuralOperatorNet, but for 3D data.
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

        # Activation
        if activation_function == "relu":
            self.activation = nn.ReLU()
        elif activation_function == "gelu":
            self.activation = nn.GELU()
        elif activation_function == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

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

        # FNO core (from Modulus, use FNO3DEncoder directly)
        self.fno = FNO3DEncoder(
            in_channels=embed_dim,
            num_fno_layers=num_layers,
            fno_layer_size=embed_dim,
            num_fno_modes=num_fno_modes,
            padding=padding,
            padding_type=padding_type,
            activation_fn=self.activation,
            coord_features=True,
        )

        # Optional MLP head
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

    def forward(self, x, condition=None, **kwargs):
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
        
        residual = x if self.big_skip else None
        x = self.encoder(x)
        if self.pos_embed_param is not None:
            x = x + self.pos_embed_param
        x = self.norm(x)
        x = self.dropout(x)
        
        # Disable autocast around FNO core for mixed precision compatibility
        with amp.autocast("cuda", enabled=False):
            x = self.fno(x)
        
        x = self.mlp(x)
        if self.big_skip and residual is not None:
            x = torch.cat([x, residual], dim=1)
        x = self.decoder(x)
        
        # Cast back to original dtype for mixed precision compatibility
        x = x.to(dtype)
        
        return x
