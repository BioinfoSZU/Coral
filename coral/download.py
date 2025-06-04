import os
import sys
import requests
import hashlib
from tqdm import tqdm
from tabulate import tabulate
from argparse import ArgumentParser
from .util import default_basecall_config, MODEL_DIR


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest, url, md5):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get('content-length', 0))
    with open(dest, "wb") as f, tqdm(
        desc=f"Downloading {os.path.basename(dest)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print("Verifying checksum...")
    if md5sum(dest) != md5:
        os.remove(dest)
        raise ValueError("Checksum mismatch!")

    print("Download complete and verified")


def add_arguments(parser: ArgumentParser):
    parser.add_argument('--list', action='store_true', default=False, help='List available models')
    parser.add_argument('--all', action='store_true', default=False, help='Download all models')
    parser.add_argument("--model", choices=default_basecall_config.keys(), default=None, help="Model to download")


def check_and_download(model):
    config = default_basecall_config[model]
    dest = os.path.join(MODEL_DIR, config['path'])
    md5_val = None
    if os.path.exists(dest):
        md5_val = md5sum(dest)
    if md5_val is not None and md5_val == config['md5']:
        print("Model {} already downloaded".format(model))
    else:
        download(dest, config['url'], config['md5'])


def run(args):
    if args.list:
        rows = [(name, info["url"], info["md5"]) for name, info in default_basecall_config.items()]
        print(tabulate(rows, headers=['Model', 'URL', "MD5"]))
        sys.exit(0)

    if args.all:
        for model in default_basecall_config.keys():
            check_and_download(model)
        sys.exit(0)

    if args.model is not None:
        check_and_download(args.model)
