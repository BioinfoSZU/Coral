import os
import torch
import torch.cuda
import torch.backends.cudnn
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
from .model import CoralModel, CoralMonotonicModel
from .encoder import get_length_after_conv
from .dataset import RNALabelDataset
from .util import linear_warmup_cosine_decay


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, working_directory, epoch):
    model.eval()
    directory = os.path.join(os.path.abspath(working_directory), 'weights')
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({
        'kmer_len': model.module.kmer_len,
        'use_cross_pos_emb': model.module.use_cross_pos_emb,
        'num_layers': model.module.decoder.num_layers,
        'state_dict': model.module.state_dict(),
    }, os.path.join(directory, 'epoch_{}.checkpoint.pth.tar'.format(epoch + 1)))


def train(epoch, train_loader, model, optimizer, scheduler, scaler, logger, args, gpu):
    model.train()
    total_loss = 0
    step_count = 0

    for step, batched_data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Epoch {}'.format(epoch)):
        (signal, targets, target_lengths) = batched_data

        global_step = epoch * len(train_loader) + step
        optimizer.zero_grad()
        signal = signal.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        target_lengths = target_lengths.cuda(gpu, non_blocking=True)

        with torch.autocast('cuda', enabled=True, dtype=torch.float16):
            loss, loss_term = model(signal, targets, target_lengths, step=global_step)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0).item()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        step_count += 1

        if logger is not None:
            logger.add_scalar('train/loss', loss.item(), global_step + 1)
            for loss_term_name, loss_term_val in loss_term.items():
                if loss_term_name == 'selected_attn':
                    if loss_term_val is not None and (global_step + 1) % 150 == 0:
                        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
                        for i in range(len(loss_term_val)):
                            sns.heatmap(loss_term_val[i], cmap="viridis", ax=axs[i // 3, i % 3], cbar=True)
                            axs[i // 3, i % 3].set_title(f'Head {i + 1} mask')
                        plt.suptitle(f"Attention Heatmaps at Step {global_step}")
                        logger.add_figure("Heatmap", fig, global_step=global_step)
                        plt.close(fig)
                    continue
                if loss_term_val is not None:
                    logger.add_scalar(f'train/{loss_term_name}', loss_term_val.item(), global_step + 1)

            logger.add_scalar('train/grad_norm', grad_norm, global_step + 1)
            logger.add_scalar('train/scaler_factor', scaler.get_scale(), global_step + 1)

            if scheduler is not None:
                scheduler_lr_list = scheduler.get_last_lr()
                if len(scheduler_lr_list) == 1:
                    logger.add_scalar('train/lr', scheduler_lr_list[0], global_step + 1)
                else:
                    for param_group_id, latest_lr in enumerate(scheduler_lr_list):
                        logger.add_scalar(f'train/lr_params{param_group_id}', latest_lr, global_step + 1)
            else:
                if len(optimizer.param_groups) == 1:
                    logger.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step + 1)
                else:
                    for param_group_id, param_group in enumerate(optimizer.param_groups):
                        logger.add_scalar(f'train/lr_params{param_group_id}', param_group['lr'], global_step + 1)
    print('GPU_{} epoch_{} train_loss = {}'.format(args.rank, epoch + 1, total_loss / step_count))


def main_worker(gpu, args):
    args.batch_size = args.batch_size // args.ngpus_per_node
    args.rank = gpu
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.ngpus_per_node, rank=args.rank)
    print('GPU rank{} use batch size {}'.format(args.rank, args.batch_size))

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        set_global_seed(args.seed)
        print('setting seed={} on cuda:{}'.format(args.seed, gpu))

    train_dataset = RNALabelDataset(
        dataset_dir=args.data, read_limit=None, is_validate=False, use_fp32=False, use_shuffle_indices=False,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.ngpus_per_node, rank=args.rank, drop_last=True
    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               drop_last=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               )

    if gpu == 0:
        print('train dataset size {}'.format(len(train_dataset)))

    logger = None
    if args.rank == 0:
        logger_path = os.path.join(os.path.abspath(args.output), 'log')
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)
        logger = SummaryWriter(logger_path)

    if args.monotonic_warmup_steps is None:
        model = CoralModel(
            dim=512, decoder_layers=args.decoder_layers, fusion_interval=3,
            kv_cache_batch_size=None, kv_cache_dtype=torch.float16,
            transformer_decoder_max_seq_len=1024,
            transformer_encoder_max_seq_len=(int(get_length_after_conv(train_dataset.signal_length)) * 2),
            enable_kv_cache=False,
            base_num=4, kmer_len=args.k,
            use_cross_pos_emb=False,
        )
    else:
        assert args.pretrained_checkpoint is not None, "Require pretrained checkpoint for monotonic regularization loss training"
        model = CoralMonotonicModel(
            dim=512, decoder_layers=args.decoder_layers, fusion_interval=3,
            kv_cache_batch_size=None, kv_cache_dtype=torch.float16,
            transformer_decoder_max_seq_len=1024,
            transformer_encoder_max_seq_len=(int(get_length_after_conv(train_dataset.signal_length)) * 2),
            enable_kv_cache=False,
            base_num=4, kmer_len=args.k,
            use_cross_pos_emb=True,
            monotonic_warmup_steps=args.monotonic_warmup_steps,
            pretrained_model=args.pretrained_checkpoint,
        )

    model = model.cuda(gpu)
    model = DDP(model, device_ids=[args.rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    scheduler_fn = linear_warmup_cosine_decay(warmup_steps=500)
    scheduler = scheduler_fn(optimizer, train_loader, args.epochs, 0)

    scaler = torch.GradScaler('cuda', enabled=True)

    for epoch in range(0, args.epochs):
        train_sampler.set_epoch(epoch)

        train(epoch=epoch,
              train_loader=train_loader,
              model=model,
              optimizer=optimizer,
              scheduler=scheduler,
              scaler=scaler,
              logger=logger,
              args=args,
              gpu=gpu
              )

        if args.rank == 0:
            save_checkpoint(model, args.output, epoch)

    if args.rank == 0:
        logger.close()

    dist.destroy_process_group()


def add_arguments(parser: ArgumentParser):
    parser.add_argument('--data', type=str, required=True, help="Training dataset directory containing rna-train.hdf5")
    parser.add_argument('--output', type=str, required=True, help="Output directory (save log and model weights)")
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:23456', help="URL specifying how to initialize the process group in multi-gpu training (default: %(default)s)")
    parser.add_argument('--ngpus-per-node', type=int, default=1, help="Number of GPUs used for training (default: %(default)s)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=128, help="Training batch size (default: %(default)s)")
    parser.add_argument('--lr', type=float, default=0.0002, help="Initial learning rate (default: %(default)s)")
    parser.add_argument('--k', type=int, default=1, help="Symbol prediction granularity, k > 1 for multi-base prediction per auto-regressive step (default: %(default)s)")
    parser.add_argument('--decoder-layers', type=int, default=12, help="Number of decoder layers (default: %(default)s)")
    parser.add_argument('--monotonic-warmup-steps', type=int, default=None, help="Warmup steps without adding monotonic regularization loss at the training start (default: %(default)s)")
    parser.add_argument('--pretrained-checkpoint', type=str, default=None, help="Pretrained model checkpoint (default: %(default)s)")
    parser.add_argument('--seed', type=int, default=40, help="Random seed for deterministic training (default: %(default)s)")


def run(args):
    if not os.path.exists(args.data):
        raise NotADirectoryError('training dataset directory is not valid')

    if not os.path.exists(args.output):
        raise NotADirectoryError('output directory is not valid')

    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.ngpus_per_node > torch.cuda.device_count():
        parser.error('--ngpus-per-node must be <= {}'.format(torch.cuda.device_count()))
    print('Current Node use {} gpus'.format(args.ngpus_per_node))

    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
