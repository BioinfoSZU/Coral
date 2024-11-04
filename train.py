import os
import argparse
import json
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from model import CoralModel
from dataset import RNALabelDataset
from pytorch_ranger import Ranger


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, working_directory, epoch, step=None):
    directory = os.path.join(os.path.abspath(working_directory), 'weights')
    if not os.path.exists(directory):
        os.makedirs(directory)
    if step is None:
        torch.save(state, os.path.join(directory, 'epoch_{}.checkpoint.pth.tar'.format(epoch)))
    else:
        torch.save(state, os.path.join(directory, 'epoch_{}_step_{}.checkpoint.pth.tar'.format(epoch, step)))


def train(epoch, train_loader, model, optimizer, logger, args, gpu):
    model.train()
    total_loss = 0
    step_count = 0

    for step, (signal, targets, target_lengths) in enumerate(train_loader):
        signal = signal.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        target_lengths = target_lengths.cuda(gpu, non_blocking=True)
        global_step = epoch * len(train_loader) + step

        loss, _ = model(signal, targets, input_lengths=None, target_lengths=target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if logger is not None and (global_step + 1) % args.print_freq == 0:
            logger.add_scalar('train/loss', loss.item(), global_step + 1)
            total_loss += loss.item()
            step_count += 1

    print('epoch_{} train_loss = {}'.format(epoch + 1, total_loss / step_count))


def evaluate(val_loader, model, logger, epoch, gpu, args):
    model.eval()
    total_loss = 0
    step_count = 0
    with (torch.no_grad()):
        for step, (signal, targets, target_lengths) in enumerate(val_loader):
            signal = signal.cuda(gpu, non_blocking=True)
            targets = targets.cuda(gpu, non_blocking=True)
            target_lengths = target_lengths.cuda(gpu, non_blocking=True)

            loss, _ = model(signal, targets, input_lengths=None, target_lengths=target_lengths)

            if logger is not None and (step + 1) % args.print_freq == 0:
                logger.add_scalar('eval_epoch{}/loss'.format(epoch + 1), loss.item(), step + 1)
                total_loss += loss.item()
                step_count += 1

    print('epoch_{} valid_loss = {}'.format(epoch + 1, total_loss / step_count))
    return total_loss / step_count


def main_worker(gpu, args):
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        set_global_seed(args.seed)

    train_dataset = RNALabelDataset(args.data, read_limit=args.limit, is_validate=False, use_fp32=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               )

    valid_dataset = RNALabelDataset(args.data, read_limit=args.valid_limit, is_validate=True, use_fp32=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.eval_batch_size,
                                               shuffle=False,
                                               drop_last=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               )

    if gpu == 0:
        print('train dataset size {}'.format(len(train_dataset)))
        print('valid dataset size {}'.format(len(valid_dataset)))

    model = CoralModel()
    model = model.cuda(gpu)
    optimizer = Ranger(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=1,
        factor=0.5,
        verbose=True,
        threshold=0.1,
        min_lr=1e-05,
    )

    logger_path = os.path.join(os.path.abspath(args.output), 'log')
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    logger = SummaryWriter(logger_path)

    for epoch in range(0, args.epochs):
        train(epoch=epoch,
              train_loader=train_loader,
              model=model,
              optimizer=optimizer,
              logger=logger,
              args=args,
              gpu=gpu
              )

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, args.output, epoch + 1)

        if (epoch + 1) % args.eval_freq == 0:
            val_loss = evaluate(valid_loader, model, logger, epoch, gpu, args)
            scheduler.step(val_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for Coral v1.0')
    parser.add_argument('data',   metavar='DATASET', help='Training dataset directory containing rna-train.hdf5 and rna-valid.hdf5')
    parser.add_argument('output', metavar='OUTPUT',  help='Output directory (save log and model weights)')
    parser.add_argument('--epochs', default=18, type=int, help='Epoch number (default: 18)')
    parser.add_argument('--batch-size', default=30, type=int, help='Batch size in training mode (default: 30)')
    parser.add_argument('--eval-batch-size', default=30, type=int, help='Batch size in evaluate mode (default: 30)')
    parser.add_argument('--lr', default=0.002, type=float, help='Initial learning rate (default: 0.002)')
    parser.add_argument('--limit', type=int, default=None, help='Reads number limit in training (default: None)')
    parser.add_argument('--valid-limit', type=int, default=30000, help='Reads number limit in validation (default: 30000)')
    parser.add_argument('--alphabet', metavar='ALPHABET', type=str, default='NACGT', help='Canonical base alphabet (default: NACGT)')
    parser.add_argument('--print-freq', default=1, type=int, help='Logging step frequency (default: 1)')
    parser.add_argument('--eval-freq',  default=1, type=int, help='Evaluation epoch frequency (default: 1)')
    parser.add_argument('--seed', default=40, type=int, help='Random seed for deterministic training (default: 40)')

    args_ = parser.parse_args()

    if not os.path.exists(args_.data):
        raise NotADirectoryError('input directory is not valid')

    if not os.path.exists(args_.output):
        os.makedirs(args_.output)

    with open(os.path.join(args_.output, 'config.json'), 'w') as f:
        json.dump(args_.__dict__, f, indent=2)

    torch.backends.cudnn.benchmark = True
    if args_.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args_.seed)

    main_worker(gpu=0, args=args_)
