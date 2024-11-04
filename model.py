import numba
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import orthogonal_
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def truncated_normal(size, dtype=torch.float32, device=None, num_resample=5):
    x = torch.empty(size + (num_resample,), dtype=dtype, device=device).normal_()
    i = ((x < 2) & (x > -2)).max(-1, keepdim=True)[1]
    return torch.clamp_(x.gather(-1, i).squeeze(-1), -2, 2)


class RNNWrapper(torch.nn.Module):
    def __init__(
            self, rnn_type, *args, reverse=False, orthogonal_weight_init=True, disable_state_bias=True,
            bidirectional=False, **kwargs
    ):
        super().__init__()
        if reverse and bidirectional:
            raise Exception("'reverse' and 'bidirectional' should not both be set to True")
        self.reverse = reverse
        self.rnn = rnn_type(*args, bidirectional=bidirectional, **kwargs)
        self.init_orthogonal(orthogonal_weight_init)
        self.init_biases()
        if disable_state_bias: self.disable_state_bias()

    @staticmethod
    def flipBatch(data, lengths):
        assert data.shape[0] == len(lengths), "Dimension Mismatch!"
        for i in range(data.shape[0]):
            data[i, :lengths[i]] = data[i, :lengths[i]].flip(dims=[0])
        return data

    def forward(self, x, input_lengths=None, hidden_states=None):
        if input_lengths is None:
            if self.reverse:
                x = x.flip(dims=[1])
            y, h = self.rnn(x, hidden_states)
            if self.reverse:
                y = y.flip(dims=[1])
            return y, h
        else:
            input_lengths = input_lengths.cpu()
            max_seq_len = x.shape[1]
            if self.reverse:
                x = self.flipBatch(x, input_lengths)
            packed_rnn_inputs = pack_padded_sequence(x, input_lengths,
                                                     batch_first=True,
                                                     enforce_sorted=False)
            y, h = self.rnn(packed_rnn_inputs, hidden_states)
            y, _ = pad_packed_sequence(y, batch_first=True, total_length=max_seq_len)
            if self.reverse:
                y = self.flipBatch(y, input_lengths)
            return y, h

    def init_biases(self, types=('bias_ih',)):
        for name, param in self.rnn.named_parameters():
            if any(k in name for k in types):
                with torch.no_grad():
                    param.set_(0.5 * truncated_normal(param.shape, dtype=param.dtype, device=param.device))

    def init_orthogonal(self, types=True):
        if not types:
            return
        if types:
            types = ('weight_ih', 'weight_hh')
        for name, x in self.rnn.named_parameters():
            if any(k in name for k in types):
                for i in range(0, x.size(0), self.rnn.hidden_size):
                    orthogonal_(x[i:i + self.rnn.hidden_size])

    def disable_state_bias(self):
        for name, x in self.rnn.named_parameters():
            if 'bias_hh' in name:
                x.requires_grad = False
                x.zero_()


class LSTMLayer(RNNWrapper):
    def __init__(self, size, insize, num_layers=1, bias=True, reverse=False):
        super().__init__(torch.nn.LSTM, size, insize, num_layers=num_layers, bias=bias, batch_first=True,
                         reverse=reverse)


class Transpose(torch.nn.Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, inputs: Tensor):
        return inputs.transpose(*self.shape)


def get_length_after_5_conv(lengths):
    lengths = torch.div(lengths + 2 * 1 - 1 * (3 - 1) - 1, 1, rounding_mode='trunc') + 1
    lengths = torch.div(lengths + 2 * 2 - 1 * (5 - 1) - 1, 1, rounding_mode='trunc') + 1
    lengths = torch.div(lengths + 2 * 5 - 1 * (10 - 1) - 1, 1, rounding_mode='trunc') + 1
    lengths = torch.div(lengths + 2 * 5 - 1 * (10 - 1) - 1, 1, rounding_mode='trunc') + 1
    lengths = torch.div(lengths + 2 * 5 - 1 * (11 - 1) - 1, 10, rounding_mode='trunc') + 1
    return lengths


class Encoder(torch.nn.Module):
    def __init__(self, encoder_dim=768, cnn_activate_type=nn.Mish):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=3 // 2,
                bias=True,
            ),
            nn.BatchNorm1d(64),
            cnn_activate_type(),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=5 // 2,
                bias=True,
            ),
            nn.BatchNorm1d(128),
            cnn_activate_type(),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=10,
                stride=1,
                padding=10 // 2,
                bias=True,
            ),
            nn.BatchNorm1d(256),
            cnn_activate_type(),

            nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=10,
                stride=1,
                padding=10 // 2,
                bias=True,
            ),
            nn.BatchNorm1d(512),
            cnn_activate_type(),

            nn.Conv1d(
                in_channels=512,
                out_channels=encoder_dim,
                kernel_size=11,
                stride=10,
                padding=11 // 2,
                bias=True,
            ),
            nn.BatchNorm1d(encoder_dim),
            cnn_activate_type(),

            Transpose(shape=(1, 2))
        )

        self.layers = torch.nn.ModuleList([
            LSTMLayer(size=encoder_dim, insize=encoder_dim, reverse=True),
            LSTMLayer(size=encoder_dim, insize=encoder_dim, reverse=False),
            LSTMLayer(size=encoder_dim, insize=encoder_dim, reverse=True),
            LSTMLayer(size=encoder_dim, insize=encoder_dim, reverse=False),
            LSTMLayer(size=encoder_dim, insize=encoder_dim, reverse=True),
        ])

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(dim=1)
        x = self.conv(x)
        for layer in self.layers:
            x, _ = layer(x, input_lengths=None)
        return x

    def variable_forward(self, x, input_lengths):
        if len(x.size()) == 2:
            x = x.unsqueeze(dim=1)
        xs = []
        B = x.size(0)
        for idx in range(B):
            xs.append(self.conv(x[idx:idx+1, :, :input_lengths[idx]]))
        input_lengths = get_length_after_5_conv(input_lengths)
        max_T, _ = torch.max(input_lengths, dim=-1)
        x = torch.cat([F.pad(tensor, (0, 0, 0, max_T - tensor.size(1), 0, 0)) for tensor in xs], dim=0)
        for layer in self.layers:
            x, _ = layer(x, input_lengths=input_lengths)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, input_dim=1025, decoder_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_cell = LSTMLayer(size=input_dim, insize=decoder_dim, num_layers=1, reverse=False)

    def forward(self, targets, target_lengths=None, hidden_states=None):
        targets = F.one_hot(targets, num_classes=self.input_dim)
        targets = targets.to(torch.float32)
        o, h = self.rnn_cell(targets, input_lengths=target_lengths, hidden_states=hidden_states)
        return o, h


def log_exclusive_cumprod(tensor, dim: int):
    tensor = tensor.roll(1, dims=dim)
    tensor.select(dim, 0).fill_(0)
    tensor = tensor.cumsum(dim)
    return tensor


def log_exclusive_cumsum(tensor, neg_inf, dim: int):
    tensor = tensor.roll(1, dims=dim)
    tensor.select(dim, 0).fill_(neg_inf)
    tensor = tensor.logcumsumexp(dim)
    return tensor


class CoralModel(torch.nn.Module):
    def __init__(
        self,
        dim=768,
        decoder_dim=512,
        cnn_activate_type=nn.Mish,
        neg_inf=-1e4,
        base_num=4,
        kmer_len=5,
        emit_prob_shift=-4,
    ):
        super().__init__()
        self.encoder = Encoder(encoder_dim=dim, cnn_activate_type=cnn_activate_type, )
        self.decoder = Decoder(decoder_dim=decoder_dim, )

        self.init_predict_layer = nn.Linear(dim + decoder_dim, 1024)
        self.other_predict_layer = nn.Linear(dim + decoder_dim, 4)

        self.W = nn.Linear(decoder_dim, dim, bias=False)
        self.scale = 1.0 / np.sqrt(dim)
        self.r = emit_prob_shift

        self.neg_inf = neg_inf
        self.base_num = base_num
        self.kmer_len = kmer_len

    def forward(
        self,
        inputs,
        targets,
        input_lengths=None,
        target_lengths=None,
    ):
        encoder_output = self.encoder(inputs)

        targets = torch.clamp(targets - 1, 0)
        B, L = targets.shape
        n = L - (self.kmer_len - 1)
        k_mer_list = sum(
            targets[:, i:n + i] * self.base_num ** (self.kmer_len - i - 1)
            for i in range(self.kmer_len)
        )

        init_true_labels = k_mer_list[:, 0]
        other_true_labels = targets[:, self.kmer_len:]

        k_mer_list = k_mer_list + 1
        k_mer_list = k_mer_list.roll(1, dims=1)
        k_mer_list[:, 0] = 0
        k_mer_list_lengths = target_lengths - (self.kmer_len - 1)

        decoder_output, _ = self.decoder(k_mer_list, target_lengths=k_mer_list_lengths, hidden_states=None)

        emit_logit = torch.bmm(self.W(decoder_output), encoder_output.transpose(1, 2))
        emit_logit = emit_logit * self.scale + self.r
        emit_prob = F.logsigmoid(emit_logit)
        emit_1_minus_prob = F.logsigmoid(-emit_logit)

        max_frame_size = encoder_output.size(1)
        max_token_size = decoder_output.size(1)

        encoder_output = encoder_output.unsqueeze(1)
        decoder_output = decoder_output.unsqueeze(2)

        init_predict_logit = self.init_predict_layer(torch.cat((
            encoder_output,
            decoder_output[:, 0, :, :].unsqueeze(1).expand(-1, -1, max_frame_size, -1)
        ), dim=-1))
        init_predict_prob = F.log_softmax(init_predict_logit, dim=-1)

        other_predict_logit = emit_prob.new_zeros([B, max_token_size - 1, max_frame_size, 4])
        for i in range(1, max_token_size):
            other_predict_logit[:, i - 1, :, :] = self.other_predict_layer(torch.cat((
                encoder_output,
                decoder_output[:, i, :, :].unsqueeze(1).expand(-1, -1, max_frame_size, -1)
            ), dim=-1)).squeeze(1)
        other_predict_prob = F.log_softmax(other_predict_logit, dim=-1)

        log_cumprod_1_minus_emit_prob = log_exclusive_cumprod(emit_1_minus_prob, dim=-1)

        log_alpha = emit_prob.new_zeros([B, max_token_size, max_frame_size])

        init_trans = init_predict_prob.gather(
            dim=-1,
            index=init_true_labels.unsqueeze(1).view(B, 1, 1, 1).expand(-1, -1, max_frame_size, -1)
        ).squeeze()
        log_alpha[:, 0, :] = init_trans + emit_prob[:, 0, :] + log_cumprod_1_minus_emit_prob[:, 0, :]

        other_trans = other_predict_prob.gather(
            dim=-1,
            index=other_true_labels.view(B, max_token_size - 1, 1, 1).expand(-1, -1, max_frame_size, -1)
        ).squeeze(-1)
        logp_prefix = other_trans + emit_prob[:, 1:, :] + log_cumprod_1_minus_emit_prob[:, 1:, :]

        def clamp_logp(x):
            return x.clamp(min=self.neg_inf, max=0)

        for i in range(1, max_token_size):
            log_alpha[:, i, :] = clamp_logp(
                logp_prefix[:, i - 1] +
                log_exclusive_cumsum(log_alpha[:, i - 1] - log_cumprod_1_minus_emit_prob[:, i],
                                     neg_inf=self.neg_inf,
                                     dim=1
                                     )
            )

        log_alpha = log_alpha[:, :, -1].gather(
            dim=1,
            index=(k_mer_list_lengths - 1).view(B, 1),
        ).view(B)

        scales = torch.tensor(
            [1.0 / k_mer_list_lengths[b] for b in range(B)],
            dtype=torch.float32, device=inputs.device
        )
        loss = - torch.mean(log_alpha * scales, dtype=torch.float32)
        return loss, emit_prob

    def get_encoder_embedding(self, inputs, input_lengths):
        encoder_output = self.encoder.variable_forward(inputs, input_lengths)
        output_lengths = get_length_after_5_conv(input_lengths)
        return encoder_output, output_lengths

    def max_decoding_batch(
        self,
        inputs,
        input_lengths,
        beam_size=30,
        duration_width=300,
        max_find_ending_count=30,
        using_len_norm=False,
        is_debug=False,
    ):
        encoder_output, time_frame_lengths = self.get_encoder_embedding(inputs, input_lengths)
        max_frame_size, _ = torch.max(time_frame_lengths, dim=0)
        encoder_output = encoder_output[:, :max_frame_size, :]
        max_frame_size = max_frame_size.item()
        max_token_size = max_frame_size

        batch_size = encoder_output.size(0)
        Q = inputs.new_full([batch_size, max_token_size, beam_size], fill_value=float("-inf"), dtype=torch.float32)
        bp = inputs.new_zeros([batch_size, max_token_size, beam_size], dtype=torch.long)
        predicts = inputs.new_zeros([batch_size, max_token_size, beam_size], dtype=torch.long)
        align_points = inputs.new_zeros([batch_size, max_token_size, beam_size], dtype=torch.long)
        ending_counting = inputs.new_zeros([batch_size], dtype=torch.long)

        decoder_input = torch.full([batch_size, 1], fill_value=0, device=inputs.device, dtype=torch.long)
        dec, (dec_h_state, dec_c_state) = self.decoder(decoder_input, hidden_states=None)

        emit_logit = torch.bmm(self.W(dec), encoder_output.transpose(1, 2))
        emit_logit = emit_logit * self.scale + self.r
        emit_prob = F.logsigmoid(emit_logit)
        emit_1_minus_prob = F.logsigmoid(-emit_logit)
        log_cumprod_1_minus_emit_prob = log_exclusive_cumprod(emit_1_minus_prob, dim=-1)

        predict_logit = self.init_predict_layer(torch.cat((
            encoder_output,
            dec.expand(-1, max_frame_size, -1)
        ), dim=-1))
        predict_prob = F.log_softmax(predict_logit, dim=-1)

        scores = predict_prob + (emit_prob + log_cumprod_1_minus_emit_prob).squeeze(1).unsqueeze(2).expand(-1, -1, 1024)

        best_score_for_k, best_token_idx_for_k = torch.max(scores, dim=-1)
        best_score_for_k.masked_fill_(
            torch.arange(max_frame_size, device=inputs.device).expand(batch_size, max_frame_size) >= time_frame_lengths.unsqueeze(1),
            float('-inf')
        )
        best_score, align_point = torch.topk(best_score_for_k, beam_size, dim=-1)
        best_token = best_token_idx_for_k[torch.arange(batch_size).unsqueeze(1), align_point]

        Q[:, 0, :] = best_score
        predicts[:, 0, :] = best_token
        align_points[:, 0, :] = align_point

        decs, (dec_h_states, dec_c_states) = self.decoder(
            (best_token + 1).reshape(batch_size * beam_size).unsqueeze(1),
            hidden_states=(
                dec_h_state.unsqueeze(2).expand(-1, -1, beam_size, -1).reshape(
                    (1, batch_size * beam_size, -1)).contiguous(),
                dec_c_state.unsqueeze(2).expand(-1, -1, beam_size, -1).reshape(
                    (1, batch_size * beam_size, -1)).contiguous()
            )
        )
        decs = decs.squeeze(1).reshape((batch_size, beam_size, -1))
        dec_h_states = dec_h_states.reshape((1, batch_size, beam_size, -1))
        dec_c_states = dec_c_states.reshape((1, batch_size, beam_size, -1))

        for i in range(1, max_token_size):
            prev_max_align_points, _ = torch.max(align_points[:, i - 1, :], dim=-1)
            ending_counting[prev_max_align_points == (time_frame_lengths - 1)] += 1
            batch_mask = (ending_counting < max_find_ending_count) & (time_frame_lengths > i)
            if (batch_mask == False).all():
                break
            current_max_frames_size, _ = torch.max(time_frame_lengths[batch_mask], dim=0)
            Q_ret, bp_ret, align_points_ret, predicts_ret, new_decs, new_dec_h_states, new_dec_c_states = self.one_step_decoding(
                encoder_output=encoder_output[batch_mask, :current_max_frames_size, :],
                time_frame_lengths=time_frame_lengths[batch_mask],
                decs=decs[batch_mask, :, :],
                dec_h_states=dec_h_states[:, batch_mask, :, :],
                dec_c_states=dec_c_states[:, batch_mask, :, :],
                Q=Q[batch_mask, i-1, :],
                predicts=predicts[batch_mask, i-1, :],
                align_points=align_points[batch_mask, i-1, :],
                duration_width=duration_width,
                beam_size=beam_size,
            )
            Q[batch_mask, i, :] = Q_ret
            bp[batch_mask, i, :] = bp_ret
            align_points[batch_mask, i, :] = align_points_ret
            predicts[batch_mask, i, :] = predicts_ret
            decs[batch_mask, :, :] = new_decs
            dec_h_states[:, batch_mask, :, :] = new_dec_h_states
            dec_c_states[:, batch_mask, :, :] = new_dec_c_states

        tgt_idx_arr = []
        beam_idx_arr = []
        for b in range(batch_size):
            target_idx_range, beam_idx_range = torch.where(align_points[b] == (time_frame_lengths[b] - 1))
            if using_len_norm:
                _, top_idx = torch.max(Q[b, target_idx_range, beam_idx_range] / (target_idx_range + 1), dim=-1)
            else:
                _, top_idx = torch.max(Q[b, target_idx_range, beam_idx_range], dim=-1)
            tgt_idx_arr.append(target_idx_range[top_idx].item())
            beam_idx_arr.append(beam_idx_range[top_idx].item())

        if is_debug:
            return np.array(tgt_idx_arr), np.array(beam_idx_arr), bp.cpu().numpy(), predicts.cpu().numpy(), Q.cpu().numpy(), align_points.cpu().numpy()

        return np.array(tgt_idx_arr), np.array(beam_idx_arr), bp.cpu().numpy(), predicts.cpu().numpy(), Q.cpu().numpy()

    def one_step_decoding(
            self,
            encoder_output,
            time_frame_lengths,
            decs,
            dec_h_states,
            dec_c_states,
            Q,
            predicts,
            align_points,
            duration_width,
            beam_size,
    ):
        batch_size = encoder_output.size(0)
        max_frame_size = encoder_output.size(1)
        duration_range = torch.arange(duration_width + 1, device=encoder_output.device).view(1, 1, duration_width + 1)
        indices = align_points.unsqueeze(-1) + duration_range
        time_frames = time_frame_lengths.view(-1, 1, 1).expand(-1, beam_size, duration_width + 1)
        mask_ = indices >= time_frames
        indices = torch.min(indices, time_frames - 1)
        batch_view = torch.arange(batch_size).view(-1, 1, 1).expand(-1, beam_size, duration_width + 1)

        encoder_output = encoder_output[batch_view, indices].reshape((batch_size * beam_size, duration_width + 1, -1))
        decs = decs.reshape((batch_size * beam_size, 1, -1))
        emit_logit = torch.bmm(self.W(decs), encoder_output.transpose(1, 2))
        emit_logit = emit_logit.reshape((batch_size, beam_size, duration_width + 1))
        emit_logit = emit_logit * self.scale + self.r
        emit_prob = F.logsigmoid(emit_logit)
        emit_1_minus_prob = F.logsigmoid(-emit_logit)
        log_cumprod_1_minus_emit_prob = log_exclusive_cumprod(emit_1_minus_prob, dim=-1)

        predict_logit = self.other_predict_layer(torch.cat((
            encoder_output,
            decs.expand(-1, (duration_width + 1), -1)
        ), dim=-1))
        predict_prob = F.log_softmax(predict_logit, dim=-1)
        predict_prob = predict_prob.reshape((batch_size, beam_size, duration_width + 1, -1))

        scores = (Q.unsqueeze(2).expand(-1, -1, duration_width + 1)
                  + emit_prob + log_cumprod_1_minus_emit_prob).unsqueeze(3).expand(-1, -1, -1, 4) + predict_prob
        scores[mask_, :] = float('-inf')
        scores = scores[:, :, 1:, :]
        indices = indices[:, :, 1:]

        best_score_for_token, best_score_for_token_idx = torch.max(scores, -1)
        scores_per_position = encoder_output.new_full(
            [max_frame_size, beam_size * duration_width], fill_value=float("-inf"),
            dtype=torch.float32
        )
        Q_ret = []
        bp_ret = []
        align_points_ret = []
        predicts_ret = []
        for b in range(batch_size):
            scores_per_position[indices[b].flatten(), torch.arange(beam_size * duration_width)] = best_score_for_token[b].flatten()
            best_score_for_beam, max_indices_flat = scores_per_position.max(dim=1)
            max_rows = torch.div(max_indices_flat, duration_width, rounding_mode='trunc')
            max_cols = max_indices_flat % duration_width
            best_score, best_align_point = torch.topk(best_score_for_beam[:time_frame_lengths[b]], beam_size, dim=-1)
            best_beam_idx = max_rows[best_align_point]
            best_token = best_score_for_token_idx[b, best_beam_idx, max_cols[best_align_point]]

            Q_ret.append(best_score.unsqueeze(0))
            bp_ret.append(best_beam_idx.unsqueeze(0))
            align_points_ret.append(best_align_point.unsqueeze(0))
            predicts_ret.append((((predicts[b, best_beam_idx] << 2) & 0b001111111100) + best_token).unsqueeze(0))
            scores_per_position.fill_(float('-inf'))

        Q_ret = torch.cat(Q_ret, dim=0)
        bp_ret = torch.cat(bp_ret, dim=0)
        align_points_ret = torch.cat(align_points_ret, dim=0)
        predicts_ret = torch.cat(predicts_ret, dim=0)
        new_decs, (new_dec_h_states, new_dec_c_states) = self.decoder(
            (predicts_ret + 1).reshape((batch_size * beam_size, 1)),
            hidden_states=(
                dec_h_states.squeeze(0)[torch.arange(batch_size).unsqueeze(1), bp_ret, :].unsqueeze(0).reshape((1, batch_size * beam_size, -1)),
                dec_c_states.squeeze(0)[torch.arange(batch_size).unsqueeze(1), bp_ret, :].unsqueeze(0).reshape((1, batch_size * beam_size, -1))
            )
        )

        new_decs = new_decs.squeeze(1).reshape((batch_size, beam_size, -1))
        new_dec_h_states = new_dec_h_states.reshape((1, batch_size, beam_size, -1))
        new_dec_c_states = new_dec_c_states.reshape((1, batch_size, beam_size, -1))
        return Q_ret, bp_ret, align_points_ret, predicts_ret, new_decs, new_dec_h_states, new_dec_c_states


@numba.njit
def decoding_backtracking(tgt_idx, b_idx, back_point, pred, Q):
    final_length = tgt_idx + 1
    seq = np.empty(final_length, dtype=np.int32)
    quality = np.empty(final_length, dtype=np.float32)
    prev_quality = None
    for i in range(final_length - 1, -1, -1):
        seq[i] = pred[tgt_idx, b_idx]
        if prev_quality is None:
            prev_quality = Q[tgt_idx, b_idx]
        else:
            quality[i + 1] = prev_quality - Q[tgt_idx, b_idx]
            prev_quality = Q[tgt_idx, b_idx]
        b_idx = back_point[tgt_idx, b_idx]
        tgt_idx -= 1
    quality[0] = prev_quality
    quality = np.exp(quality)
    read_mean_quality = -10 * math.log10(np.mean(1 - quality))

    magic = 0b0000000011
    base_seq = np.empty(len(seq) + 4, dtype=np.int32)
    base_seq[0] = (seq[0] >> 8) & magic
    base_seq[1] = (seq[0] >> 6) & magic
    base_seq[2] = (seq[0] >> 4) & magic
    base_seq[3] = (seq[0] >> 2) & magic
    base_seq[4] = (seq[0] >> 0) & magic
    for i, kmer in enumerate(seq[1:]):
        base_seq[i + 5] = (kmer & magic)
    return base_seq, read_mean_quality
