import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Tuple, Dict
import torchtune
import torchtune.models.llama3_1
from torchtune.modules.model_fusion._fusion_layers import FusionLayer
from torchtune.modules.attention_utils import _MaskType, _sdpa_or_flex_attention
from torchtune.modules.kv_cache import KVCache


def kmer_encode_batched(tokens, lengths, k, alphabet_size=4):
    B, T = tokens.shape
    num_groups = math.ceil(T / k)
    pad_size = num_groups * k - T

    padded_tokens = torch.cat([tokens, torch.zeros(B, pad_size, dtype=tokens.dtype, device=tokens.device)], dim=1)
    groups = padded_tokens.view(B, num_groups, k)

    new_lengths = torch.ceil(lengths.float() / k).long()

    positions = torch.arange(num_groups * k, device=tokens.device).view(1, num_groups, k)
    valid_mask = positions < lengths.view(B, 1, 1)

    n_valid = valid_mask.sum(dim=2)

    powers = torch.tensor([alphabet_size ** i for i in range(k)], device=tokens.device, dtype=torch.long).view(1, 1, k)

    encoding_val = (groups * powers).sum(dim=2)

    offset = torch.zeros_like(encoding_val)
    for j in range(k):
        offset[torch.where(torch.eq(n_valid, k - j))] = (sum(alphabet_size ** i for i in range(k, k - j, -1)) + 1)

    encoded = encoding_val + offset
    return encoded.view(B, -1), new_lengths


def decode_kmer_token(encoded_val, k=3, alphabet_size=4, SOS_token=0, EOS_token=None):
    assert EOS_token is not None

    if encoded_val == EOS_token:
        return []

    if encoded_val == SOS_token:
        return []

    encoded_val = encoded_val - 1
    _range_start = 0
    _range_end = 0
    ret = []
    for n_valid in range(k, 0, -1):
        _range_end = sum([
            alphabet_size ** i
            for i in range(n_valid, k + 1)
        ])
        if _range_start <= encoded_val < _range_end:
            for m in range(n_valid):
                ret.append(encoded_val % alphabet_size)
                encoded_val = encoded_val // alphabet_size
            break
        _range_start = _range_end
    return ret


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: Optional[nn.Module] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
        add_monotonic_bias: bool = False,
    ):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q and k norm must be set together")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal
        self.add_monotonic_bias = add_monotonic_bias
        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.pos_embeddings = pos_embeddings
        self._attention_call = _sdpa_or_flex_attention()
        self.cache_enabled = False

    def setup_cache(
        self, batch_size: int, dtype: torch.dtype, max_seq_len: int
    ) -> None:
        if self.kv_cache is not None:
            logger.warning(
                "Key value caches are already setup. You cannot call ``setup_caches()`` twice. Skipping."
            )
        else:
            self.kv_cache = KVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )
            self.cache_enabled = True

    def reset_cache(self):
        if self.kv_cache is None:
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )
        self.kv_cache.reset()

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        b, s_x, _ = x.shape
        s_y = y.shape[1] if y is not None else 0

        q = self.q_proj(x)
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)
        q = q.transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)

        if y is None:
            if self.kv_cache is None or not self.cache_enabled:
                raise ValueError(
                    "Must provide y input or use kv_cache to enable streaming decoding"
                )
            k = self.kv_cache.k_cache
            v = self.kv_cache.v_cache
        else:
            k = self.k_proj(y)
            v = self.v_proj(y)
            k = k.view(b, s_y, -1, self.head_dim)
            v = v.view(b, s_y, -1, self.head_dim)
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=None)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.k_norm is not None:
                k = self.k_norm(k)
            if self.kv_cache is not None and self.cache_enabled:
                k, v = self.kv_cache.update(k, v)

        if self.num_heads != self.num_kv_heads:
            expand_shape = (b, self.num_kv_heads, q_per_kv, -1, self.head_dim)
            k = k.unsqueeze(2).expand(expand_shape).flatten(1, 2)
            v = v.unsqueeze(2).expand(expand_shape).flatten(1, 2)

        attn_scores = None
        alignment_positions = None
        if self.add_monotonic_bias:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
            time_stamps = torch.arange(attn_scores.size(-1), device=attn_scores.device, dtype=attn_scores.dtype).view(1, 1, 1, -1)
            alignment_positions = torch.sum(attn_weights * time_stamps, dim=-1)

        output = self._attention_call(
            q,
            k,
            v,
            mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )

        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)
        return self.output_proj(output), attn_scores, alignment_positions


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        attn: MultiHeadCrossAttention,
        mlp: nn.Module,
        *,
        ca_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        ca_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.ca_norm = ca_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.ca_scale = ca_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        self.attn.setup_cache(batch_size, dtype, encoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        return self.attn.kv_cache is not None

    def caches_are_enabled(self) -> bool:
        return self.attn.cache_enabled

    def reset_cache(self):
        self.attn.reset_cache()

    def _skip_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            mask = ~mask
        else:
            mask = torch.isneginf(mask)
        mask = torch.all(mask, dim=-1, keepdim=True)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        empty_cache = not self.caches_are_enabled() or self.attn.kv_cache.size == 0
        if encoder_input is None and empty_cache:
            return x

        skip_mask = self._skip_mask(encoder_mask)
        if encoder_mask is not None:
            encoder_mask = encoder_mask.masked_fill(skip_mask, True)

        attn_out, attn_scores, alignment_positions = self.attn(
            self.ca_norm(x),
            encoder_input,
            mask=encoder_mask,
            input_pos=input_pos,
        )

        if skip_mask is not None:
            attn_out = attn_out.masked_fill(skip_mask, 0)

        h = self.ca_scale(attn_out) + x

        mlp_out = self.mlp(self.mlp_norm(h))
        if skip_mask is not None:
            mlp_out = mlp_out.masked_fill(skip_mask, 0)

        out = h + self.mlp_scale(mlp_out)

        return out, attn_scores, alignment_positions


class CoralDecoder(torch.nn.Module):
    def __init__(
        self,
        kv_cache_batch_size,
        kv_cache_dtype,
        enable_kv_cache,
        decoder_max_seq_len,
        encoder_max_seq_len,
        vocab_size,
        decoder_dim=512,
        num_layers=12,
        fusion_interval=3,
        norm_eps=1e-05,
        use_cross_pos_emb=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.fusion_interval = fusion_interval

        self.cross_attn_layer_list = []

        decoder_layers = nn.ModuleList()
        for layer_idx in range(1, num_layers + 1):
            decoder_layer = torchtune.modules.TransformerSelfAttentionLayer(
                attn=torchtune.modules.MultiHeadAttention(
                    embed_dim=decoder_dim,
                    num_heads=8,
                    num_kv_heads=8,
                    head_dim=64,
                    q_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                    k_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                    v_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                    output_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                    pos_embeddings=torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE(
                        64, max_seq_len=decoder_max_seq_len, base=10000, scale_factor=8,
                    ),
                    q_norm=None,
                    k_norm=None,
                    max_seq_len=decoder_max_seq_len,
                    is_causal=True,
                    attn_dropout=0.0,
                ),
                mlp=torchtune.modules.FeedForward(
                    gate_proj=nn.Linear(decoder_dim, 2048),
                    down_proj=nn.Linear(2048, decoder_dim),
                    up_proj=nn.Linear(decoder_dim, 2048),
                ),
                sa_norm=torchtune.modules.RMSNorm(decoder_dim, eps=norm_eps),
                mlp_norm=torchtune.modules.RMSNorm(decoder_dim, eps=norm_eps),
            )
            if layer_idx % fusion_interval == 0:
                self.cross_attn_layer_list.append(layer_idx)
                curr_num_heads = 8
                curr_head_dim = decoder_dim // curr_num_heads
                xattn_layer = CrossAttentionLayer(
                    attn=MultiHeadCrossAttention(
                        embed_dim=decoder_dim,
                        num_heads=curr_num_heads,
                        num_kv_heads=curr_num_heads,
                        head_dim=curr_head_dim,
                        q_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                        k_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                        v_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                        output_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                        q_norm=None if use_cross_pos_emb else torchtune.modules.RMSNorm(dim=curr_head_dim, eps=norm_eps),
                        k_norm=None if use_cross_pos_emb else torchtune.modules.RMSNorm(dim=curr_head_dim, eps=norm_eps),
                        pos_embeddings=torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE(
                            curr_head_dim, max_seq_len=max(decoder_max_seq_len, encoder_max_seq_len), base=10000, scale_factor=8,
                        ) if use_cross_pos_emb else None,
                        max_seq_len=encoder_max_seq_len,
                        is_causal=False,
                        attn_dropout=0.0,
                        add_monotonic_bias=True if layer_idx == num_layers else False,
                    ),
                    mlp=torchtune.modules.FeedForward(
                        gate_proj=nn.Linear(decoder_dim, 2048),
                        down_proj=nn.Linear(2048, decoder_dim),
                        up_proj=nn.Linear(decoder_dim, 2048),
                    ),
                    ca_norm=torchtune.modules.RMSNorm(dim=decoder_dim),
                    mlp_norm=torchtune.modules.RMSNorm(dim=decoder_dim),
                    ca_scale=torchtune.modules.tanh_gate.TanhGate(),
                    mlp_scale=torchtune.modules.tanh_gate.TanhGate(),
                )
                fusion_layer = FusionLayer(layer=decoder_layer, fusion_layer=xattn_layer, fusion_first=True)
                decoder_layers.append(fusion_layer)
            else:
                decoder_layers.append(decoder_layer)

        self.decoder = torchtune.modules.TransformerDecoder(
            tok_embeddings=nn.Embedding(num_embeddings=vocab_size, embedding_dim=decoder_dim),
            layers=decoder_layers,
            max_seq_len=decoder_max_seq_len,
            num_heads=8,
            head_dim=64,
            norm=torchtune.modules.RMSNorm(decoder_dim, eps=norm_eps),
            output=nn.Linear(decoder_dim, vocab_size, bias=False),
        )

        self.encoder_max_seq_len = encoder_max_seq_len
        self.decoder_max_seq_len = decoder_max_seq_len
        self.curr_pos = 0
        self.enable_kv_cache = enable_kv_cache

        if enable_kv_cache:
            for layer in self.decoder.layers:
                layer.setup_caches(
                    batch_size=kv_cache_batch_size,
                    dtype=kv_cache_dtype,
                    encoder_max_seq_len=self.encoder_max_seq_len,
                    decoder_max_seq_len=self.decoder_max_seq_len,
                )

            self.register_buffer(
                "masks",
                torch.tril(torch.ones(self.decoder_max_seq_len, self.decoder_max_seq_len, dtype=torch.bool)).unsqueeze(0),
                persistent=False,
            )
            self.register_buffer(
                "encoder_masks",
                torch.ones(self.decoder_max_seq_len, self.encoder_max_seq_len, dtype=torch.bool).unsqueeze(0),
                persistent=False,
            )
            self.register_buffer(
                "input_pos",
                torch.arange(0, self.decoder_max_seq_len).unsqueeze(0),
                persistent=False,
            )

    def reset_kv_cache(self):
        if not self.decoder.caches_are_enabled():
            raise RuntimeError("kv_cache is not enabled")
        self.decoder.reset_caches()
        self.curr_pos = 0

    def custom_decoder_forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        ret_attn_scores = []
        ret_alignment_pos = []
        h = self.decoder.tok_embeddings(tokens)
        assert isinstance(self.decoder.layers, nn.ModuleList)
        for i, layer in enumerate(self.decoder.layers):
            layer_idx = (i + 1)
            if layer_idx in self.cross_attn_layer_list:
                h, attn_scores, alignment_positions = layer.fusion_layer(
                    h,
                    encoder_input=encoder_input,
                    encoder_mask=encoder_mask,
                    input_pos=input_pos,
                )
                if attn_scores is not None:
                    ret_attn_scores.append(attn_scores)

                if alignment_positions is not None:
                    ret_alignment_pos.append(alignment_positions)

                h = layer.layer(
                    h,
                    mask=mask,
                    encoder_input=encoder_input,
                    encoder_mask=encoder_mask,
                    input_pos=input_pos,
                )
            else:
                h = layer(
                    h,
                    mask=mask,
                    encoder_input=encoder_input,
                    encoder_mask=encoder_mask,
                    input_pos=input_pos,
                )

        logits = self.decoder.unembed(h)
        return logits, torch.cat(ret_attn_scores, dim=1), torch.cat(ret_alignment_pos, dim=1)

    def forward(self, targets, encoder_inputs, encoder_mask=None):
        if self.enable_kv_cache:
            with torchtune.modules.disable_kv_cache(self.decoder):
                logits, attn_scores, align_pos = self.custom_decoder_forward(targets, encoder_input=encoder_inputs, encoder_mask=encoder_mask)
        else:
            logits, attn_scores, align_pos = self.custom_decoder_forward(targets, encoder_input=encoder_inputs, encoder_mask=encoder_mask)
        return logits, attn_scores, align_pos

    def reorder_kv_cache(self, updated_beams_order):
        for module in self.decoder.layers:
            if isinstance(module, torchtune.modules.TransformerSelfAttentionLayer):
                layer = module
            else:  # FusionLayer
                layer = module.layer
            layer.attn.kv_cache.k_cache[:, :, :self.curr_pos, :] = layer.attn.kv_cache.k_cache[updated_beams_order, :, :self.curr_pos, :]
            layer.attn.kv_cache.v_cache[:, :, :self.curr_pos, :] = layer.attn.kv_cache.v_cache[updated_beams_order, :, :self.curr_pos, :]
