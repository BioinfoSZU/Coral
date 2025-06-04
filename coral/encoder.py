import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache, partial
from torch import Tensor
from typing import Optional, List, Tuple, Dict
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_func
        from flash_attn.layers.rotary import RotaryEmbedding
        from flash_attn.modules.mlp import GatedMlp
        from flash_attn.ops.triton.layer_norm import RMSNorm
except (ImportError, RuntimeError) as e:
    print(f"Error: failed to import flash_attn: {e}", file=sys.stderr)
    sys.exit(1)


class Transpose(torch.nn.Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, inputs: Tensor):
        return inputs.transpose(*self.shape)


def deepnorm_params(depth):
    alpha = round((2*depth)**0.25, 7)
    beta = round((8*depth)**(-1/4), 7)
    return alpha, beta


@lru_cache(maxsize=2)
def sliding_window_mask(seq_len, window, device):
    band = torch.full((seq_len, seq_len), fill_value=1.0)
    band = torch.triu(band, diagonal=-window[0])
    band = band * torch.tril(band, diagonal=window[1])
    band = band.to(torch.bool).to(device)
    return band


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.Wqkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=True)
        self.rotary_emb = RotaryEmbedding(self.head_dim, interleaved=False)

    @staticmethod
    def attn_func(qkv):
        if torch.cuda.get_device_capability(qkv.device)[0] >= 8 and (torch.is_autocast_enabled() or qkv.dtype == torch.half):
            attn_output = flash_attn_qkvpacked_func(qkv, causal=False, window_size=(127, 128))
        else:
            q, k, v = torch.chunk(qkv.permute(0, 2, 3, 1, 4), chunks=3, dim=1)
            mask = sliding_window_mask(qkv.shape[1], (127, 128), q.device)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attn_output = attn_output.permute(0, 1, 3, 2, 4)
        return attn_output

    def forward(self, x):
        N, T, _ = x.shape
        qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)
        qkv = self.rotary_emb(qkv)
        attn_output = MultiHeadAttention.attn_func(qkv).reshape(N, T, self.d_model)
        out = self.out_proj(attn_output)
        return out


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, deepnorm_alpha, deepnorm_beta):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
        )
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.deepnorm_beta = deepnorm_beta
        self.reset_parameters()

    def reset_parameters(self):
        db = self.deepnorm_beta
        d_model = self.d_model
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[2*d_model:], gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[:2*d_model], gain=1)

    def forward(self, x):
        x = self.norm1(self.self_attn(x), self.deepnorm_alpha * x)
        x = self.norm2(self.ff(x), self.deepnorm_alpha * x)
        return x


class LinearUpsample(torch.nn.Module):
    def __init__(self, d_model, scale_factor):
        super().__init__()
        self.d_model = d_model
        self.scale_factor = scale_factor
        self.linear = torch.nn.Linear(d_model, self.scale_factor * d_model)

    def forward(self, src):
        N, L, E = src.shape
        h = self.linear(src).reshape(N, self.scale_factor * L, E)
        return h


class ConvolutionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


def get_length_after_conv_template(length, paddings, kernels, strides):
    for (padding, kernel, stride) in zip(paddings, kernels, strides):
        length = (length + 2 * padding - 1 * (kernel - 1) - 1) // stride + 1
    return length


get_length_after_conv = partial(
    get_length_after_conv_template,
    paddings=[2, 2, 4, 4, 2],
    kernels=[5, 5, 9, 9, 5],
    strides=[1, 1, 3, 2, 2]
)


class CoralEncoder(torch.nn.Module):
    def __init__(self, encoder_dim=512, num_layers=18):
        super().__init__()
        self.conv_modules = torch.nn.ModuleList([
            ConvolutionModule(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=5 // 2,
                bias=True,
            ),
            ConvolutionModule(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=5 // 2,
                bias=True,
            ),
            ConvolutionModule(
                in_channels=64,
                out_channels=128,
                kernel_size=9,
                stride=3,
                padding=9 // 2,
                bias=True,
            ),
            ConvolutionModule(
                in_channels=128,
                out_channels=128,
                kernel_size=9,
                stride=2,
                padding=9 // 2,
                bias=True,
            ),
            ConvolutionModule(
                in_channels=128,
                out_channels=encoder_dim,
                kernel_size=5,
                stride=2,
                padding=5 // 2,
                bias=True,
            ),
            Transpose(shape=(1, 2))
        ])

        self.num_layers = num_layers
        alpha, beta = deepnorm_params(num_layers)
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(
                d_model=encoder_dim,
                nhead=8,
                dim_feedforward=2048,
                deepnorm_alpha=alpha,
                deepnorm_beta=beta,
            )
            for _ in range(num_layers)
        ])
        self.upsample = LinearUpsample(d_model=encoder_dim, scale_factor=2)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(dim=1)
        for layer in self.conv_modules:
            x = layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.upsample(x)
        return x
