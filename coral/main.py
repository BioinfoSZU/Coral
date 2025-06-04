from argparse import ArgumentParser
from .version import __version__
from . import download, basecall, train, accuracy


def main():
    parser = ArgumentParser(prog="python -m coral", description="Nanopore Direct-RNA sequencing basecaller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download the pretrained model checkpoints")
    download.add_arguments(download_parser)
    download_parser.set_defaults(func=download.run)

    basecall_parser = subparsers.add_parser("basecall", help="Basecalling")
    basecall.add_arguments(basecall_parser)
    basecall_parser.set_defaults(func=basecall.run)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train.add_arguments(train_parser)
    train_parser.set_defaults(func=train.run)

    accuracy_parser = subparsers.add_parser("accuracy", help="Compute the read accuracy given a SAM file")
    accuracy.add_arguments(accuracy_parser)
    accuracy_parser.set_defaults(func=accuracy.run)

    parser.add_argument('--version', action='version', version=f'Coral {__version__}')
    args = parser.parse_args()
    args.func(args)
