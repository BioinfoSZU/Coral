import os
import h5py
import pod5
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import contextlib
from tqdm import tqdm
from tabulate import tabulate
from glob import glob
from pathlib import Path
from Bio import SeqIO
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from .decoder import decode_kmer_token
from .model import CoralModel
from .util import *
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    from ont_fast5_api.fast5_interface import get_fast5_file


def signal_generator(f5_paths, pod5_r: pod5.DatasetReader, is_pod5, _requires=None, _called=None):
    if is_pod5:
        for read_record in pod5_r.reads(selection=_requires):
            _rid = read_record.read_id
            if _called is not None and _rid in _called:
                yield _rid, None
            else:
                yield _rid, read_record.signal_pa.astype(np.float32)
    else:
        for _filename in f5_paths:
            with get_fast5_file(_filename, 'r') as _fh:
                for _rid in _fh.get_read_ids():
                    if _requires is not None and _rid not in _requires:
                        continue
                    if _called is not None and _rid in _called:
                        yield _rid, None
                    else:
                        _read = _fh.get_read(_rid)
                        _raw = _read.handle[_read.raw_dataset_name][:]
                        _channel_info = _read.handle[_read.global_key + 'channel_id'].attrs
                        _offset = int(_channel_info['offset'])
                        _scaling = _channel_info['range'] / _channel_info['digitisation']
                        yield _rid, np.array(_scaling * (_raw + _offset), dtype=np.float32)


def custom_cross_attn_forward(
    module,
    x,
    y=None,
    mask=None,
    input_pos=None,
    is_final_cross_layer=False,
):
    # x has shape [b, s_x, d]
    # y has shape [b, s_y, d]
    b, s_x, _ = x.shape
    s_y = y.shape[1] if y is not None else 0

    # q has shape [b, s_x, num_heads * head_dim]
    q = module.q_proj(x)

    # number of queries per key/value
    q_per_kv = module.num_heads // module.num_kv_heads
    q = q.view(b, s_x, module.num_kv_heads * q_per_kv, module.head_dim)

    # Apply positional embeddings
    if module.pos_embeddings is not None:
        q = module.pos_embeddings(q, input_pos=input_pos)

    # [b, n_h, s_x, h_d]
    q = q.transpose(1, 2)

    # Normalize q
    if module.q_norm is not None:
        q = module.q_norm(q)

    if y is None:
        if module.kv_cache is None or not module.cache_enabled:
            raise ValueError(
                "Must provide y input or use kv_cache to enable streaming decoding"
            )
        k = module.kv_cache.k_cache
        v = module.kv_cache.v_cache
    else:
        # Update k and v shape, positional embeddings, and normalization

        # k,v shape [b, s_y, num_kv_heads * head_dim]
        k = module.k_proj(y)
        v = module.v_proj(y)

        # Apply positional embeddings
        # k,v shape: [b, s_y, n_kv, h_d]
        k = k.view(b, s_y, -1, module.head_dim)
        v = v.view(b, s_y, -1, module.head_dim)
        if module.pos_embeddings is not None:
            k = module.pos_embeddings(k, input_pos=None)

        # k,v shape: [b, n_kv, s_y, h_d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Normalize k
        if module.k_norm is not None:
            k = module.k_norm(k)

        # Update key-value cache
        if module.kv_cache is not None and module.cache_enabled:
            k, v = module.kv_cache.update(k, v)

    # k,v shape: [b, n_kv, s_y, h_d] -> [b, n_h, s_y, h_d]
    if module.num_heads != module.num_kv_heads:
        expand_shape = (b, module.num_kv_heads, q_per_kv, -1, module.head_dim)
        k = k.unsqueeze(2).expand(expand_shape).flatten(1, 2)
        v = v.unsqueeze(2).expand(expand_shape).flatten(1, 2)

    if is_final_cross_layer:
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (module.head_dim ** -0.5)  # [b, n_h, s_x, s_y]

        # Note: mask.shape is [B, s_x, s_y]
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        time_stamps = torch.arange(attn_weights.size(-1), device=attn_weights.device, dtype=attn_weights.dtype).view(1, 1, 1, -1)

        alignment_positions = torch.sum(attn_weights * time_stamps, dim=-1).to(torch.long)  # [b, n_h, s_x]

        output = torch.matmul(attn_weights, v)  # [b, n_h, s_x, h_d]

        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)

        return module.output_proj(output), attn_weights, alignment_positions
    else:
        output = module._attention_call(
            q,
            k,
            v,
            mask=mask,
            dropout_p=module.attn_dropout if module.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)
        return module.output_proj(output), None, None


def custom_cross_layer_forward(
    module,
    x,
    encoder_input=None,
    encoder_mask=None,
    input_pos=None,
    is_final_cross_layer=False,
):
    empty_cache = not module.caches_are_enabled() or module.attn.kv_cache.size == 0
    if encoder_input is None and empty_cache:
        return x

    attn_out, attn_weights, alignment_positions = custom_cross_attn_forward(
        module.attn,
        x=module.ca_norm(x),
        y=encoder_input,
        mask=encoder_mask,
        input_pos=input_pos,
        is_final_cross_layer=is_final_cross_layer,
    )

    h = module.ca_scale(attn_out) + x

    mlp_out = module.mlp(module.mlp_norm(h))

    out = h + module.mlp_scale(mlp_out)

    return out, attn_weights, alignment_positions


def custom_inference(
    module,
    tokens,
    encoder_input,
    final_enc_masks,
):
    curr_in_token_size = tokens.size(1)
    if curr_in_token_size == 1:
        curr_masks = module.masks[:, module.curr_pos, None, :]
        curr_enc_masks = module.encoder_masks[:, module.curr_pos, None, :]
        curr_input_pos = module.input_pos[:, module.curr_pos]
    else:
        curr_masks = module.masks[:, :curr_in_token_size, :]
        curr_enc_masks = module.encoder_masks[:, :curr_in_token_size, :]
        curr_input_pos = module.input_pos[:, :curr_in_token_size]

    h = module.decoder.tok_embeddings(tokens)

    assert isinstance(module.decoder.layers, nn.ModuleList)

    final_attn_weights = None
    final_alignment_positions = None
    for i, layer in enumerate(module.decoder.layers):
        layer_idx = (i + 1)
        if layer_idx in module.cross_attn_layer_list:
            is_final_cross_layer = (layer_idx == module.cross_attn_layer_list[-1])

            h, attn_weights, alignment_positions = custom_cross_layer_forward(
                layer.fusion_layer,
                x=h,
                encoder_input=encoder_input,
                encoder_mask=curr_enc_masks if not is_final_cross_layer else final_enc_masks,
                input_pos=curr_input_pos,
                is_final_cross_layer=is_final_cross_layer,
            )
            if attn_weights is not None:
                final_attn_weights = attn_weights
            if alignment_positions is not None:
                final_alignment_positions = alignment_positions

            h = layer.layer(
                h,
                mask=curr_masks,
                encoder_input=encoder_input,
                encoder_mask=curr_enc_masks,
                input_pos=curr_input_pos,
            )
        else:
            h = layer(
                h,
                mask=curr_masks,
                encoder_input=encoder_input,
                encoder_mask=curr_enc_masks,
                input_pos=curr_input_pos,
            )

    logits = module.decoder.unembed(h)

    ret_logits = logits[:, -1, None, :]

    ret_attn_weights = final_attn_weights[:, :, -1, None, :]  # B x H x 1 x T

    ret_align_pos = final_alignment_positions[:, :, -1, None]  # B x H x 1

    module.curr_pos += curr_in_token_size

    return ret_logits, ret_attn_weights, ret_align_pos


@dataclass
class Output:
    seqs: List
    aligns: List
    called_num: int
    num_chunks: int
    sig_len: int
    stub: int


class Basecaller:
    def print_config(self):
        rows = [
            ("model_path", self.weight_path),
            ("chunksize", self.chunksize),
            ("feature_dimension", self.dim),
            ("head_dimension", self.head_dim),
            ("bin_len", self.kmer_len),
            ("decoder_layers", self.decoder_layers),
            ("beam_size", self.beam_size),
        ]
        print(tabulate(rows, headers=['ModelConfig', 'Value']))

    def parse_config(self):
        self.chunksize = default_basecall_config[self.model_name]['chunksize']
        self.stride = default_basecall_config[self.model_name]['stride']
        self.overlap = default_basecall_config[self.model_name]['overlap']
        self.dim = default_basecall_config[self.model_name]['dim']
        self.head_num = default_basecall_config[self.model_name]['head_num']
        self.head_dim = self.dim // self.head_num
        self.kmer_len = default_basecall_config[self.model_name]['kmer_len']
        self.decoder_layers = default_basecall_config[self.model_name]['decoder_layers']
        self.fusion_interval = default_basecall_config[self.model_name]['fusion_interval']
        self.weight_path = str(os.path.join(MODEL_DIR, default_basecall_config[self.model_name]['path']))
        self.use_cross_pos_emb = default_basecall_config[self.model_name]['use_cross_pos_emb']
        self.first_max_seq_len = default_basecall_config[self.model_name]['first_max_seq_len']
        self.max_seq_len = default_basecall_config[self.model_name]['max_seq_len']
        self.disable_second_phase = False
        if self.first_max_seq_len == self.max_seq_len:
            self.disable_second_phase = True
        self.encoder_max_seq_len = default_basecall_config[self.model_name]['encoder_max_seq_len']
        self.beam_size = default_basecall_config[self.model_name]['beam_size']
        self.norm_parameters = default_basecall_config[self.model_name]['norm_parameters']
        self.decoding_options = default_basecall_config[self.model_name]['decoding_options']
        self.qual_trim = self.decoding_options['qual_trim']
        self.strict_region_len = self.decoding_options['strict_region_len']

    def create_model(self):
        self.model = CoralModel(
            dim=self.dim,
            decoder_layers=self.decoder_layers,
            fusion_interval=self.fusion_interval,
            kv_cache_batch_size=(self.batch_size * self.beam_size),
            kv_cache_dtype=torch.float16,
            transformer_decoder_max_seq_len=self.max_seq_len,
            transformer_encoder_max_seq_len=self.encoder_max_seq_len,
            enable_kv_cache=True,
            kmer_len=self.kmer_len,
            use_cross_pos_emb=self.use_cross_pos_emb,
        )
        self.model = self.model.cuda(self.gpu)
        if os.path.isfile(self.weight_path):
            with safe_globals():
                checkpoint = torch.load(self.weight_path, map_location=f'cuda:{self.gpu}')
            self.model.load_state_dict(checkpoint)
        else:
            raise RuntimeError('No checkpoint found at {}'.format(self.weight_path))
        self.model.eval()
        self.SOS_token = 0
        self.EOS_token = self.model.vocab_size - 1

    def get_fast5_meta(self):
        sequencing_kit = None
        fast5_type = None
        with h5py.File(self.fast5_paths[0], 'r') as f:
            try:
                if 'UniqueGlobalKey' in f.keys():  # single-read fast5
                    sequencing_kit = f['UniqueGlobalKey']['context_tags'].attrs['sequencing_kit'].decode()
                    fast5_type = "single-read"
                else:  # maybe multi-reads fast5
                    for _key in f.keys():
                        sequencing_kit = f[_key]['context_tags'].attrs['sequencing_kit'].decode()
                        break
                    fast5_type = "multi-read"
            except KeyError as e:
                print('WARNING: No sequencing_kit tag in fast5 file: {}'.format(e))

        if fast5_type == "multi-read" and self.parse_fast5_meta:
            fast5_cnt = 0
            for filename in tqdm(self.fast5_paths, desc='Read multi-fast5 meta'):
                with get_fast5_file(filename, 'r') as f5_fh:
                    fast5_cnt += len(f5_fh.get_read_ids())
        else:
            fast5_cnt = len(self.fast5_paths)

        return fast5_cnt, sequencing_kit, fast5_type

    def setup_input_pipeline(self):
        self.require_reads = None
        if self.reads_file is not None:
            self.require_reads = set()
            with open(self.reads_file, 'r') as f:
                for line in f:
                    self.require_reads.add(line.strip())

        self.fast5_paths = None
        self.pod5_reader = None
        if not self.use_pod5:
            self.fast5_paths = [Path(x) for x in glob(self.fast5_dir + "/" + "**/*.fast5", recursive=True)]
            if len(self.fast5_paths) <= 0:
                raise RuntimeError('There are no fast5 in directory {}'.format(self.fast5_dir))
            read_cnt, sequencing_kit, fast5_type = self.get_fast5_meta()
            if sequencing_kit is not None:
                print('Sequencing kit:', sequencing_kit)
            if fast5_type is not None:
                print('Fast5 type:', fast5_type)
            print('Number of fast5 reads:', read_cnt)
        else:
            self.pod5_reader = pod5.DatasetReader(self.fast5_dir)  # when use pod5, the directory is path of pod5 file
            read_cnt = self.pod5_reader.num_reads
            print('Number of pod5 reads:', read_cnt)
        self.pbar = tqdm(total=read_cnt if self.require_reads is None else min(len(self.require_reads), read_cnt))

    def setup_output_pipeline(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.fasta_path = os.path.join(self.output_dir, f'{self.output_name}.fasta')
        self.err_path = os.path.join(self.output_dir, f'{self.output_name}_fail_reads.txt')

        self.called_db = None
        if os.path.exists(self.fasta_path):
            self.called_db = set()
            for item in SeqIO.parse(self.fasta_path, 'fasta'):
                self.called_db.add(str(item.id))
            print('Existed called sequences:', len(self.called_db))
            self.output_fa = open(self.fasta_path, 'a')
        else:
            self.output_fa = open(self.fasta_path, 'w')

        self.num_failed_calls = 0
        if os.path.exists(self.err_path):
            self.err_f = open(self.err_path, 'a')
        else:
            self.err_f = open(self.err_path, 'w')

    def setup_pipeline(self):
        self.setup_input_pipeline()
        self.setup_output_pipeline()

        self.signal_gen = signal_generator(
            self.fast5_paths,
            self.pod5_reader,
            self.use_pod5,
            self.require_reads,
            self.called_db
        )

        self.output_db = {}
        self.global_chunks: Optional[torch.Tensor] = None
        self.global_ranges = []
        self.second_global_chunks: Optional[torch.Tensor] = None
        self.second_global_ranges = []

        self.predicts_cache: Optional[torch.Tensor] = None
        self.predict_probs_cache: Optional[torch.Tensor] = None
        self.candidate_lens_cache: Optional[torch.Tensor] = None
        self.Q_cache: Optional[torch.Tensor] = None
        self.custom_enc_masks_cache: Optional[torch.Tensor] = None
        self.path_cache: Optional[torch.Tensor] = None

    def __init__(
        self,
        model_name,
        fast5_dir,
        output_dir,
        gpu,
        batch_size,
        beam_size=None,
        output_name='basecaller',
        use_pod5=False,
        parse_fast5_meta=True,
        reads_file=None,
        verbose=True,
    ):
        self.model_name = model_name
        self.fast5_dir = fast5_dir
        self.output_dir = output_dir
        self.gpu = gpu
        self.batch_size = batch_size
        self.output_name = output_name
        self.use_pod5 = use_pod5
        self.parse_fast5_meta = parse_fast5_meta
        self.reads_file = reads_file
        self.parse_config()
        if beam_size is not None:
            self.beam_size = beam_size
        if verbose:
            self.print_config()
        self.create_model()
        self.setup_pipeline()

    def basecalling(
        self,
        input_signals,
        max_token_size,
        predicts=None,
        predict_probs=None,
        candidate_lens=None,
        Q=None,
        custom_enc_masks=None,
        path=None,
    ):
        device = input_signals.device
        max_full_kmer = 4 ** self.kmer_len

        encoder_output = self.model.encoder(input_signals)
        batch_range_tensor = torch.arange(self.batch_size, device=device).unsqueeze(1) * self.beam_size
        batch_beam_range_tensor = torch.arange(self.batch_size * self.beam_size, device=device).unsqueeze(1)
        frames_tensor = torch.arange(self.encoder_max_seq_len, dtype=torch.long, device=device)

        restart_inference = False
        if predicts is not None:
            restart_inference = True

        if candidate_lens is None:
            candidate_lens = torch.full([self.batch_size * self.beam_size], fill_value=int(10000), device=device, dtype=torch.long)

        if predicts is None:
            decoder_input = torch.full([self.batch_size * self.beam_size, 1], fill_value=self.SOS_token, device=device, dtype=torch.long)

            custom_enc_masks = torch.ones(self.batch_size * self.beam_size, 1, self.encoder_max_seq_len, device=device, dtype=torch.bool)

            dec_logits, _, head_alignments = custom_inference(
                self.model.decoder,
                tokens=decoder_input,
                encoder_input=torch.repeat_interleave(encoder_output, self.beam_size, dim=0),
                final_enc_masks=custom_enc_masks,
            )

            path = torch.min(head_alignments[:, :, -1], dim=1)[0].unsqueeze(1)  # [B*beam_size, 1]

            dec_logits = dec_logits.squeeze(1)  # [B*beam_size, vocab_size]
            dec_logits = torch.nn.functional.log_softmax(dec_logits, dim=-1)
            dec_probs = torch.exp(dec_logits)

            topk_scores, topk_indices = torch.topk(dec_logits, self.beam_size)
            Q = topk_scores.view(self.batch_size, self.beam_size, self.beam_size)[:, 0, :].reshape(self.batch_size * self.beam_size)
            predicts = topk_indices.view(self.batch_size, self.beam_size, self.beam_size)[:, 0, :].reshape(self.batch_size * self.beam_size, 1)
            predict_probs = dec_probs[batch_beam_range_tensor, topk_indices].view(
                self.batch_size, self.beam_size, self.beam_size
            )[:, 0, :].reshape(self.batch_size * self.beam_size, 1)

            candidate_lens[torch.where(
                torch.eq(predicts[:, 0], self.EOS_token) &
                torch.eq(candidate_lens, 10000)
            )] = 0
            candidate_lens[torch.where(
                torch.gt(predicts[:, 0], max_full_kmer) &
                torch.eq(candidate_lens, 10000)
            )] = 1

        start_token_index = predicts.size(1)
        for i in range(start_token_index, max_token_size):
            stop_indices = torch.where(
                torch.eq(predicts[:, i - 1], self.EOS_token) |
                torch.gt(predicts[:, i - 1], max_full_kmer)
            )

            curr_extend_len = torch.zeros(
                [self.batch_size * self.beam_size, 1],
                device=device,
                dtype=torch.long
            )
            curr_extend_len[
                torch.where(
                    torch.ge(path[:, -1, None], self.encoder_max_seq_len - self.strict_region_len)
                )
            ] = -1

            custom_enc_masks_new = (
                frames_tensor.view(1, -1) >= (path[:, -1, None] - curr_extend_len).clamp(min=0, max=(self.encoder_max_seq_len - 1))
            ).unsqueeze(1)

            custom_enc_masks = torch.cat([
                custom_enc_masks,
                custom_enc_masks_new
            ], dim=1)

            if restart_inference and (i == start_token_index):
                dec_logits, _, head_alignments_new = custom_inference(
                    self.model.decoder,
                    tokens=torch.cat([
                        torch.full([self.batch_size * self.beam_size, 1], fill_value=self.SOS_token, device=device, dtype=torch.long),
                        predicts
                    ], dim=1),
                    encoder_input=torch.repeat_interleave(encoder_output, self.beam_size, dim=0),
                    final_enc_masks=custom_enc_masks,
                )
            else:
                dec_logits, _, head_alignments_new = custom_inference(
                    self.model.decoder,
                    tokens=predicts[:, i - 1].view(self.batch_size * self.beam_size, 1),
                    encoder_input=None,
                    final_enc_masks=custom_enc_masks[:, -1, None, :],
                )

            dec_logits = dec_logits.squeeze(1)  # [B*beam_size, vocab_size]

            if stop_indices[0].numel() > 0:
                dec_logits[stop_indices[0], :-1] = float('-inf')  # mask all tokens (exclude EOS) to -inf logits

            dec_logits = torch.nn.functional.log_softmax(dec_logits, dim=-1)
            dec_probs = torch.exp(dec_logits)
            dec_logits = Q.unsqueeze(1) + dec_logits

            if self.beam_size > 1:
                topk_scores, topk_indices = torch.topk(dec_logits.view(self.batch_size, -1), self.beam_size)  # B x beam_size
                parent_beam_indices = topk_indices // self.model.vocab_size
                parent_beam_indices = parent_beam_indices + batch_range_tensor
                parent_beam_indices = parent_beam_indices.view(self.batch_size * self.beam_size)
                next_tokens = topk_indices % self.model.vocab_size

                # update Q
                Q = topk_scores.view(self.batch_size * self.beam_size)

                # reorder candidate_lens
                candidate_lens = candidate_lens[parent_beam_indices]

                # reorder predict
                predicts = torch.cat([predicts[parent_beam_indices, :], next_tokens.view(self.batch_size * self.beam_size, 1)], dim=1)

                # updated predict prob
                predict_probs = torch.cat([
                    predict_probs[parent_beam_indices, :],
                    dec_probs[parent_beam_indices, next_tokens.view(self.batch_size * self.beam_size)].unsqueeze(1)
                ], dim=1)

                # (B * beam_size) x (i + 1)
                path = torch.cat([
                    path[parent_beam_indices, :],
                    torch.min(head_alignments_new[parent_beam_indices, :, -1], dim=1)[0].unsqueeze(1)
                ], dim=1)
            else:
                parent_beam_indices = None
                topk_scores, topk_indices = torch.max(dec_logits.view(self.batch_size, -1), dim=1)
                next_tokens = topk_indices.unsqueeze(1)
                Q = topk_scores
                predicts = torch.cat([predicts, next_tokens], dim=1)
                predict_probs = torch.cat([
                    predict_probs,
                    dec_probs[torch.arange(self.batch_size, device=device), next_tokens.view(self.batch_size)].unsqueeze(1)
                ], dim=1)
                path = torch.cat([
                    path,
                    torch.min(head_alignments_new[:, :, -1], dim=1)[0].unsqueeze(1)
                ], dim=1)

            candidate_lens[torch.where(
                torch.eq(predicts[:, i], self.EOS_token) &
                torch.eq(candidate_lens, 10000)
            )] = i
            candidate_lens[torch.where(
                torch.gt(predicts[:, i], max_full_kmer) &
                torch.eq(candidate_lens, 10000)
            )] = (i + 1)
            candidate_lens[torch.where(
                torch.eq(path[:, i], self.encoder_max_seq_len - 1) &
                torch.eq(candidate_lens, 10000)
            )] = (i + 1)

            if torch.all(candidate_lens.view(self.batch_size, self.beam_size)[:, 0] < 10000).item():
                break

            # reorder kv-cache
            if self.beam_size > 1:
                self.model.decoder.reorder_kv_cache(updated_beams_order=parent_beam_indices)

        self.model.decoder.reset_kv_cache()

        candidate_lens_ret = candidate_lens.clone().view(self.batch_size, self.beam_size)

        non_finish_batches = torch.where(torch.eq(candidate_lens.view(self.batch_size, self.beam_size)[:, 0], 10000))[0]
        if non_finish_batches.numel() > 0:
            candidate_lens.view(self.batch_size, self.beam_size)[non_finish_batches, 0] = max_token_size

        Q = Q.view(self.batch_size, self.beam_size)
        predicts = predicts.view(self.batch_size, self.beam_size, -1)
        predict_probs = predict_probs.view(self.batch_size, self.beam_size, -1)
        candidate_lens = candidate_lens.view(self.batch_size, self.beam_size)
        custom_enc_masks = custom_enc_masks.view(self.batch_size, self.beam_size, -1, self.encoder_max_seq_len)
        path = path.view(self.batch_size, self.beam_size, -1)

        ret_seqs = []
        ret_qualities = []
        ret_paths = []

        for batch_id in range(self.batch_size):
            if candidate_lens[batch_id, 0].item() == 0:
                ret_seqs.append(None)
                ret_qualities.append(None)
                ret_paths.append(None)
                continue

            ret_seqs.append(predicts[batch_id, 0, :candidate_lens[batch_id, 0]].cpu())
            ret_acces = predict_probs[batch_id, 0, :candidate_lens[batch_id, 0]]
            ret_quality = -10 * torch.log10(1 - ret_acces.clamp(max=1 - 1e-5))  # 最低错误率控制在 0.00001 (质量值最高50, +33对应S)
            ret_qualities.append(ret_quality.round().int().cpu())
            ret_paths.append(path[batch_id, 0, :candidate_lens[batch_id, 0]].cpu())

        return (
            ret_seqs,
            ret_qualities,
            ret_paths,
            non_finish_batches.cpu(),
            predicts.cpu(),
            predict_probs.cpu(),
            candidate_lens_ret.cpu(),
            Q.cpu(),
            custom_enc_masks.cpu(),
            path.cpu()
        )

    def chunk_decoding(
        self,
        signal_chunks,
        chunk_ranges,
        max_token_size,
        predicts=None,
        predict_probs=None,
        candidate_lens=None,
        Q=None,
        custom_enc_masks=None,
        path=None,
        is_first_stage=False,
    ):
        with torch.no_grad():
            with torch.autocast('cuda', enabled=True):
                seqs, qualities, align_paths, un_finish_batches, _predicts, _predict_probs, _candidate_lens, _Q, _custom_enc_masks, _path = self.basecalling(
                    input_signals=signal_chunks.cuda(self.gpu, non_blocking=True),
                    max_token_size=max_token_size,
                    predicts=predicts.cuda(self.gpu, non_blocking=True) if predicts is not None else None,
                    predict_probs=predict_probs.cuda(self.gpu, non_blocking=True) if predict_probs is not None else None,
                    candidate_lens=candidate_lens.cuda(self.gpu, non_blocking=True) if candidate_lens is not None else None,
                    Q=Q.cuda(self.gpu, non_blocking=True) if Q is not None else None,
                    custom_enc_masks=custom_enc_masks.cuda(self.gpu, non_blocking=True) if custom_enc_masks is not None else None,
                    path=path.cuda(self.gpu, non_blocking=True) if path is not None else None,
                )

        reserve_batch_indices = []
        for b in range(len(chunk_ranges)):
            un_finish = False
            if len(un_finish_batches) > 0 and b in un_finish_batches:
                un_finish = True

            if not self.disable_second_phase and is_first_stage and un_finish:
                reserve_batch_indices.append(b)
                continue

            seq_id, c_id = chunk_ranges[b]
            seq_tokens: Optional[torch.Tensor] = seqs[b]
            seq_qualities: Optional[torch.Tensor] = qualities[b]
            seq_alignments: Optional[torch.Tensor] = align_paths[b]

            if seq_tokens is not None:
                qs = seq_qualities.to(torch.float32).mean().item()
                seq_bases = []
                for val in seq_tokens:
                    seq_bases.append(
                        ''.join(['ACGT'[ch] for ch in decode_kmer_token(val, k=self.kmer_len, SOS_token=self.SOS_token, EOS_token=self.EOS_token)])
                    )
            else:
                qs = 0
                seq_bases = []

            if qs >= self.qual_trim:
                self.output_db[seq_id].seqs[c_id] = seq_bases
                self.output_db[seq_id].aligns[c_id] = seq_alignments

            self.output_db[seq_id].called_num += 1
            if self.output_db[seq_id].called_num >= self.output_db[seq_id].num_chunks:
                final_seq = ""
                if self.output_db[seq_id].num_chunks == 1:
                    if self.output_db[seq_id].seqs[0] is not None:
                        final_seq = ''.join(self.output_db[seq_id].seqs[0])
                else:
                    if self.kmer_len == 1:
                        semi_overlap = self.overlap // 2
                        _frame_st, _frame_en = semi_overlap // self.stride, self.encoder_max_seq_len - semi_overlap // self.stride
                        _first_frame_en = (self.output_db[seq_id].stub + semi_overlap) // self.stride if (self.output_db[seq_id].stub > 0) else _frame_en
                        for cid in range(self.output_db[seq_id].num_chunks):
                            if self.output_db[seq_id].seqs[cid] is None:
                                continue
                            if cid == 0:
                                _st = -1
                                _en = _first_frame_en if self.output_db[seq_id].seqs[cid + 1] is not None else self.encoder_max_seq_len
                            elif cid == (self.output_db[seq_id].num_chunks - 1):
                                _st = _frame_st if self.output_db[seq_id].seqs[cid - 1] is not None else -1
                                _en = self.encoder_max_seq_len
                            else:
                                _st = _frame_st if self.output_db[seq_id].seqs[cid - 1] is not None else -1
                                _en = _frame_en if self.output_db[seq_id].seqs[cid + 1] is not None else self.encoder_max_seq_len

                            _st_index = torch.where(self.output_db[seq_id].aligns[cid] > _st)[0]
                            _en_index = torch.where(self.output_db[seq_id].aligns[cid] < _en)[0]
                            if _st_index.numel() > 0 and _en_index.numel() > 0:
                                _st_index = _st_index[0]
                                _en_index = _en_index[-1] + 1
                                if _st_index < _en_index:
                                    final_seq += ''.join(self.output_db[seq_id].seqs[cid][_st_index: _en_index])
                    else:
                        overlap_st_arr = [None for _ in range(self.output_db[seq_id].num_chunks)]
                        overlap_en_arr = [None for _ in range(self.output_db[seq_id].num_chunks)]

                        for cid in range(self.output_db[seq_id].num_chunks - 1):
                            if cid == 0:
                                overlap_region = (self.chunksize - self.output_db[seq_id].stub) // self.stride if (self.output_db[seq_id].stub > 0) else self.overlap // self.stride
                            else:
                                overlap_region = self.overlap // self.stride

                            first_chunk_en, second_chunk_st = None, None
                            if self.output_db[seq_id].seqs[cid] is not None and self.output_db[seq_id].seqs[cid + 1] is not None:
                                _index = torch.where(torch.le(self.output_db[seq_id].aligns[cid + 1], overlap_region))[0]
                                if _index.numel() > 0:
                                    _index = _index[-1] + 1
                                    query_seq = ''.join(self.output_db[seq_id].seqs[cid])
                                    ref_seq = ''.join(self.output_db[seq_id].seqs[cid + 1][:_index])
                                    first_chunk_en, second_chunk_st = alignment_process(query_seq, ref_seq, seq_id)

                            overlap_en_arr[cid] = first_chunk_en
                            overlap_st_arr[cid + 1] = second_chunk_st

                        for cid in range(self.output_db[seq_id].num_chunks):
                            if self.output_db[seq_id].seqs[cid] is None:
                                continue
                            seq_str = ''.join(self.output_db[seq_id].seqs[cid])
                            _s_st = 0 if overlap_st_arr[cid] is None else overlap_st_arr[cid]
                            _s_en = len(seq_str) if overlap_en_arr[cid] is None else overlap_en_arr[cid]
                            final_seq += seq_str[_s_st: _s_en]

                if len(final_seq) > 0:
                    self.output_fa.write(f'>{seq_id}\n{final_seq[::-1]}\n')
                    self.output_fa.flush()
                else:
                    self.err_f.write(f'{seq_id}\n')
                    self.err_f.flush()
                    self.num_failed_calls += 1

                _ = self.output_db.pop(seq_id, None)
                self.pbar.update()

        if len(reserve_batch_indices) > 0:
            if self.second_global_chunks is None:
                self.second_global_chunks = signal_chunks[reserve_batch_indices]
            else:
                self.second_global_chunks = torch.cat([self.second_global_chunks, signal_chunks[reserve_batch_indices]], dim=0)

            if self.predicts_cache is None:
                self.predicts_cache = _predicts[reserve_batch_indices]
            else:
                self.predicts_cache = torch.cat([self.predicts_cache, _predicts[reserve_batch_indices]], dim=0)

            if self.predict_probs_cache is None:
                self.predict_probs_cache = _predict_probs[reserve_batch_indices]
            else:
                self.predict_probs_cache = torch.cat([self.predict_probs_cache, _predict_probs[reserve_batch_indices]], dim=0)

            if self.candidate_lens_cache is None:
                self.candidate_lens_cache = _candidate_lens[reserve_batch_indices]
            else:
                self.candidate_lens_cache = torch.cat([self.candidate_lens_cache, _candidate_lens[reserve_batch_indices]], dim=0)

            if self.Q_cache is None:
                self.Q_cache = _Q[reserve_batch_indices]
            else:
                self.Q_cache = torch.cat([self.Q_cache, _Q[reserve_batch_indices]], dim=0)

            if self.custom_enc_masks_cache is None:
                self.custom_enc_masks_cache = _custom_enc_masks[reserve_batch_indices]
            else:
                self.custom_enc_masks_cache = torch.cat([self.custom_enc_masks_cache, _custom_enc_masks[reserve_batch_indices]], dim=0)

            if self.path_cache is None:
                self.path_cache = _path[reserve_batch_indices]
            else:
                self.path_cache = torch.cat([self.path_cache, _path[reserve_batch_indices]], dim=0)

            for _b_idx in reserve_batch_indices:
                self.second_global_ranges.append(chunk_ranges[_b_idx])

    def run(self, ):
        for read_id, scaled in self.signal_gen:
            if scaled is None:
                self.pbar.update()
                continue

            shift, scale = normalisation(scaled, norm_params=self.norm_parameters)
            trimmed_samples = trim(scaled, threshold=scale * 2.4 + shift)
            signal = (scaled[trimmed_samples:] - shift) / scale
            signal = torch.from_numpy(signal).to(torch.float32)
            T = len(signal)
            if T < self.chunksize:
                self.err_f.write(f'{read_id}\n')
                self.err_f.flush()
                self.num_failed_calls += 1
                self.pbar.update()
                continue

            stub = (T - self.overlap) % (self.chunksize - self.overlap)
            chunks = signal[stub:].unfold(0, self.chunksize, self.chunksize - self.overlap)
            if stub > 0:
                chunks = torch.cat([signal[None, :self.chunksize], chunks], dim=0)

            if self.global_chunks is None:
                self.global_chunks = chunks
            else:
                self.global_chunks = torch.cat([self.global_chunks, chunks], dim=0)

            num_chunk = len(chunks)
            if read_id not in self.output_db:
                self.output_db[read_id] = Output(
                    seqs=[None for _ in range(num_chunk)],
                    aligns=[None for _ in range(num_chunk)],
                    called_num=0,
                    num_chunks=num_chunk,
                    sig_len=T,
                    stub=stub,
                )

            self.global_ranges.append((read_id, 0))
            for _chunk_id in range(1, num_chunk - 1):
                self.global_ranges.append((read_id, _chunk_id))
            if num_chunk > 1:
                self.global_ranges.append((read_id, num_chunk - 1))

            if self.global_chunks is not None and len(self.global_chunks) >= self.batch_size:
                self.chunk_decoding(
                    signal_chunks=self.global_chunks[:self.batch_size],
                    chunk_ranges=self.global_ranges[:self.batch_size],
                    max_token_size=self.first_max_seq_len,
                    is_first_stage=True,
                )
                self.global_chunks = self.global_chunks[self.batch_size:]
                self.global_ranges = self.global_ranges[self.batch_size:]

            if self.second_global_chunks is not None and len(self.second_global_chunks) >= self.batch_size:
                self.chunk_decoding(
                    signal_chunks=self.second_global_chunks[:self.batch_size],
                    chunk_ranges=self.second_global_ranges[:self.batch_size],

                    predicts=self.predicts_cache[:self.batch_size].view(self.batch_size * self.beam_size, *self.predicts_cache.shape[2:]),
                    predict_probs=self.predict_probs_cache[:self.batch_size].view(self.batch_size * self.beam_size, *self.predict_probs_cache.shape[2:]),
                    candidate_lens=self.candidate_lens_cache[:self.batch_size].view(self.batch_size * self.beam_size, *self.candidate_lens_cache.shape[2:]),
                    Q=self.Q_cache[:self.batch_size].view(self.batch_size * self.beam_size, *self.Q_cache.shape[2:]),
                    custom_enc_masks=self.custom_enc_masks_cache[:self.batch_size].view(self.batch_size * self.beam_size, *self.custom_enc_masks_cache.shape[2:]),
                    path=self.path_cache[:self.batch_size].view(self.batch_size * self.beam_size, *self.path_cache.shape[2:]),

                    max_token_size=self.max_seq_len,
                    is_first_stage=False,
                )
                self.second_global_chunks = self.second_global_chunks[self.batch_size:]
                self.second_global_ranges = self.second_global_ranges[self.batch_size:]
                self.predicts_cache = self.predicts_cache[self.batch_size:]
                self.predict_probs_cache = self.predict_probs_cache[self.batch_size:]
                self.candidate_lens_cache = self.candidate_lens_cache[self.batch_size:]
                self.Q_cache = self.Q_cache[self.batch_size:]
                self.custom_enc_masks_cache = self.custom_enc_masks_cache[self.batch_size:]
                self.path_cache = self.path_cache[self.batch_size:]

        if self.global_chunks is not None and len(self.global_chunks) > 0:
            self.chunk_decoding(
                signal_chunks=get_padded_zero_tensor(self.global_chunks, self.batch_size),
                chunk_ranges=self.global_ranges,
                max_token_size=self.max_seq_len,
                is_first_stage=False,
            )

        if self.second_global_chunks is not None and len(self.second_global_chunks) > 0:
            self.chunk_decoding(
                signal_chunks=get_padded_zero_tensor(self.second_global_chunks, self.batch_size),
                chunk_ranges=self.second_global_ranges,
                max_token_size=self.max_seq_len,

                predicts=get_padded_zero_tensor(self.predicts_cache, self.batch_size).view(self.batch_size * self.beam_size, *self.predicts_cache.shape[2:]),
                predict_probs=get_padded_zero_tensor(self.predict_probs_cache, self.batch_size).view(self.batch_size * self.beam_size, *self.predict_probs_cache.shape[2:]),
                candidate_lens=get_padded_zero_tensor(self.candidate_lens_cache, self.batch_size).view(self.batch_size * self.beam_size, *self.candidate_lens_cache.shape[2:]),
                Q=get_padded_zero_tensor(self.Q_cache, self.batch_size).view(self.batch_size * self.beam_size, *self.Q_cache.shape[2:]),
                custom_enc_masks=get_padded_zero_tensor(self.custom_enc_masks_cache, self.batch_size).view(self.batch_size * self.beam_size, *self.custom_enc_masks_cache.shape[2:]),
                path=get_padded_zero_tensor(self.path_cache, self.batch_size).view(self.batch_size * self.beam_size, *self.path_cache.shape[2:]),

                is_first_stage=False,
            )

        if len(self.output_db) > 0:
            print(f'Error: {len(self.output_db)} reads = {self.output_db.keys()} incomplete.')

    def clear(self):
        self.pbar.close()
        self.output_fa.close()
        self.err_f.close()
        print('Failed called reads:', self.num_failed_calls)
