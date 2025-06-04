import torch
from .encoder import CoralEncoder
from .decoder import CoralDecoder, kmer_encode_batched, decode_kmer_token
from .util import safe_globals


def compute_monotonic_bias_loss(alignments, target_length, encoder_max_seq_len, margin=1.0, loss_weight=1.0):
    B, N, L = alignments.shape
    device = alignments.device

    diff_pos = alignments[:, :, :-1] - alignments[:, :, 1:] + margin

    valid_length_mask = (torch.arange(L - 1, device=device).unsqueeze(0) < (target_length - 1).unsqueeze(1)).float()
    valid_length_mask = valid_length_mask.unsqueeze(1)

    loss = torch.nn.functional.relu(diff_pos / encoder_max_seq_len) * valid_length_mask

    loss = loss.sum(dim=[0, 2])
    valid_tokens_cnt = valid_length_mask.sum(dim=[0, 2])

    loss = loss / valid_tokens_cnt

    return loss_weight * loss.mean()


class CoralModel(torch.nn.Module):
    def __init__(
        self,
        dim=512,
        decoder_layers=12,
        fusion_interval=3,
        kv_cache_batch_size=None,
        kv_cache_dtype=torch.float32,
        transformer_decoder_max_seq_len=None,
        transformer_encoder_max_seq_len=None,
        enable_kv_cache=False,
        base_num=4,
        kmer_len=5,
        use_cross_pos_emb=True,
    ):
        super().__init__()
        self.encoder = CoralEncoder(encoder_dim=dim, num_layers=18)
        self.kmer_len = kmer_len
        self.base_num = base_num

        self.vocab_size = 0
        for i in range(kmer_len + 1):
            self.vocab_size += (base_num ** i)
        self.vocab_size += 1

        self.use_cross_pos_emb = use_cross_pos_emb
        self.decoder = CoralDecoder(
            kv_cache_batch_size=kv_cache_batch_size,
            kv_cache_dtype=kv_cache_dtype,
            enable_kv_cache=enable_kv_cache,
            decoder_max_seq_len=transformer_decoder_max_seq_len,
            encoder_max_seq_len=transformer_encoder_max_seq_len,
            vocab_size=self.vocab_size,
            decoder_dim=dim,
            num_layers=decoder_layers,
            fusion_interval=fusion_interval,
            use_cross_pos_emb=use_cross_pos_emb
        )

    def forward(self, inputs, targets, target_lengths, step=None):
        encoder_output = self.encoder(inputs)
        batch_size = inputs.size(0)
        kmer_targets, kmer_target_lengths = kmer_encode_batched(targets - 1, target_lengths, k=self.kmer_len, alphabet_size=self.base_num)
        max_seq_len = kmer_target_lengths.max().item()

        input_tokens = kmer_targets.new_full([batch_size, max_seq_len + 1], fill_value=(self.vocab_size - 1), dtype=torch.long)
        target_tokens = kmer_targets.new_full([batch_size, max_seq_len + 1], fill_value=(self.vocab_size - 1), dtype=torch.long)

        input_tokens[:, 0] = 0  # SOS token
        for b in range(batch_size):
            length = kmer_target_lengths[b].item()
            input_tokens[b, 1:length + 1] = kmer_targets[b, :length]
            target_tokens[b, :length] = kmer_targets[b, :length]

        logits, attn_scores, align_pos = self.decoder.forward(
            targets=input_tokens,
            encoder_inputs=encoder_output,
            encoder_mask=None,
        )

        coral_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.decoder.vocab_size),
            target_tokens.view(-1),
            reduction='none',
        )
        seq_mask = torch.arange(max_seq_len + 1, device=kmer_target_lengths.device).unsqueeze(0) < (kmer_target_lengths + 1).unsqueeze(1)
        seq_mask = seq_mask.view(-1)
        coral_loss = coral_loss * seq_mask
        coral_loss = coral_loss.sum() / seq_mask.sum()

        batch_id = torch.randint(0, batch_size, (1,)).item()
        selected_attn = attn_scores[batch_id, :, :(kmer_target_lengths + 1)[batch_id], :].clone().detach().cpu()
        selected_attn = torch.nn.functional.softmax(selected_attn, dim=-1).numpy()

        loss = coral_loss
        return loss, {
            'coral_loss': coral_loss,
            'selected_attn': selected_attn,
        }


class CoralMonotonicModel(torch.nn.Module):
    def __init__(
        self,
        dim=512,
        decoder_layers=12,
        fusion_interval=3,
        kv_cache_batch_size=None,
        kv_cache_dtype=torch.float32,
        transformer_decoder_max_seq_len=None,
        transformer_encoder_max_seq_len=None,
        enable_kv_cache=False,
        base_num=4,
        kmer_len=5,
        use_cross_pos_emb=True,
        monotonic_warmup_steps=None,
        pretrained_model=None,
    ):
        super().__init__()
        self.encoder = CoralEncoder(encoder_dim=dim, num_layers=18)

        tmp_model = None
        if pretrained_model is not None:
            with safe_globals():
                tmp_checkpoint = torch.load(pretrained_model, map_location='cpu')
            tmp_model = CoralModel(
                dim=dim,
                decoder_layers=tmp_checkpoint['num_layers'],
                fusion_interval=fusion_interval,
                kv_cache_batch_size=kv_cache_batch_size,
                kv_cache_dtype=kv_cache_dtype,
                transformer_encoder_max_seq_len=transformer_encoder_max_seq_len,
                transformer_decoder_max_seq_len=transformer_decoder_max_seq_len,
                enable_kv_cache=False,
                base_num=base_num,
                kmer_len=tmp_checkpoint['kmer_len'],
                use_cross_pos_emb=tmp_checkpoint['use_cross_pos_emb'],
            )
            tmp_model.load_state_dict(tmp_checkpoint['state_dict'])
            print('Loaded pretrained model from {}'.format(pretrained_model))

        if tmp_model is not None:
            self.encoder.load_state_dict(tmp_model.encoder.state_dict())

        self.kmer_len = kmer_len
        self.base_num = base_num
        self.vocab_size = 0
        for i in range(kmer_len + 1):
            self.vocab_size += (base_num ** i)
        self.vocab_size += 1

        self.use_cross_pos_emb = use_cross_pos_emb
        self.monotonic_warmup_steps = monotonic_warmup_steps
        self.decoder = CoralDecoder(
            kv_cache_batch_size=kv_cache_batch_size,
            kv_cache_dtype=kv_cache_dtype,
            enable_kv_cache=enable_kv_cache,
            decoder_max_seq_len=transformer_decoder_max_seq_len,
            encoder_max_seq_len=transformer_encoder_max_seq_len,
            vocab_size=self.vocab_size,
            decoder_dim=dim,
            num_layers=decoder_layers,
            fusion_interval=fusion_interval,
            use_cross_pos_emb=use_cross_pos_emb,
        )
        if tmp_model is not None:
            if self.kmer_len == tmp_model.kmer_len:
                self.decoder.decoder.tok_embeddings.load_state_dict(tmp_model.decoder.decoder.tok_embeddings.state_dict())
                self.decoder.decoder.norm.load_state_dict(tmp_model.decoder.decoder.norm.state_dict())
                self.decoder.decoder.output.load_state_dict(tmp_model.decoder.decoder.output.state_dict())

            for layer_idx in range(1, decoder_layers + 1):
                if layer_idx not in self.decoder.cross_attn_layer_list:
                    self.decoder.decoder.layers[layer_idx - 1].load_state_dict(
                        tmp_model.decoder.decoder.layers[layer_idx - 1].state_dict()
                    )
                else:
                    if self.kmer_len != tmp_model.kmer_len:
                        self.decoder.decoder.layers[layer_idx - 1].load_state_dict(
                            tmp_model.decoder.decoder.layers[layer_idx - 1].state_dict()
                        )

    def forward(self, inputs, targets, target_lengths, step=None):
        encoder_output = self.encoder(inputs)

        batch_size = inputs.size(0)
        kmer_targets, kmer_target_lengths = kmer_encode_batched(targets - 1, target_lengths, k=self.kmer_len, alphabet_size=self.base_num)
        max_seq_len = kmer_target_lengths.max().item()

        input_tokens = kmer_targets.new_full([batch_size, max_seq_len + 1], fill_value=(self.vocab_size - 1), dtype=torch.long)
        target_tokens = kmer_targets.new_full([batch_size, max_seq_len + 1], fill_value=(self.vocab_size - 1), dtype=torch.long)

        input_tokens[:, 0] = 0  # SOS token
        for b in range(batch_size):
            length = kmer_target_lengths[b].item()
            input_tokens[b, 1:length + 1] = kmer_targets[b, :length]
            target_tokens[b, :length] = kmer_targets[b, :length]

        logits, attn_scores, align_pos = self.decoder.forward(
            targets=input_tokens,
            encoder_inputs=encoder_output,
            encoder_mask=None,
        )

        coral_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.decoder.vocab_size),
            target_tokens.view(-1),
            reduction='none',
        )
        seq_mask = torch.arange(max_seq_len + 1, device=kmer_target_lengths.device).unsqueeze(0) < (kmer_target_lengths + 1).unsqueeze(1)
        seq_mask = seq_mask.view(-1)
        coral_loss = coral_loss * seq_mask
        coral_loss = coral_loss.sum() / seq_mask.sum()

        monotonic_bias_loss = compute_monotonic_bias_loss(
            alignments=align_pos,
            target_length=(kmer_target_lengths + 1),
            encoder_max_seq_len=encoder_output.size(1),
            margin=float(self.kmer_len),
            loss_weight=1.0,
        )

        batch_id = torch.randint(0, batch_size, (1,)).item()
        selected_attn = attn_scores[batch_id, :, :(kmer_target_lengths + 1)[batch_id], :].clone().detach().cpu()  # only select the final heads
        selected_attn = torch.nn.functional.softmax(selected_attn, dim=-1).numpy()

        if self.monotonic_warmup_steps is not None and step > self.monotonic_warmup_steps:
            loss = (coral_loss + monotonic_bias_loss)
        else:
            loss = coral_loss

        return loss, {
            'coral_loss': coral_loss,
            'selected_attn': selected_attn,
            'monotonic_bias_loss': monotonic_bias_loss,
        }
