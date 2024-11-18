## Coral: a dual context-aware basecaller for nanopore direct RNA sequencing

### Download and install

#### System dependencies
- NVIDIA driver version >= 450.80.02
- CUDA Toolkit >= 11.3

Coral v1.0 can be installed on Linux and has been tested on Ubuntu 22.04 with RTX 3090 GPU.

#### Install from conda
We recommend users install Coral using [conda](https://www.anaconda.com/download/success) command, the installation typically takes around ten minutes.
```shell
git clone https://github.com/BioinfoSZU/Coral.git
cd Coral
wget -P model https://zenodo.org/records/14033197/files/weights.pth.tar # download model checkpoint
echo "c4dd28b7d72da91cfb27e944f511b851 model/weights.pth.tar" | md5sum -c --quiet  # check md5sum
conda create -n coral python==3.7.11
conda activate coral
conda install -c bioconda minimap2==2.17
pip install --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt 
```

### Usage
- Basecaller options:
```text
usage: coral.py [-h] [--prefix PREFIX] [--gpu GPU]
                [--read-id-path READ_ID_PATH] [--chunksize CHUNKSIZE]
                [--batch-size BATCH_SIZE] [--model MODEL] [--version]
                input_dir output_dir

Coral: a basecaller for nanopore direct-RNA sequencing

positional arguments:
  input_dir             Directory containing fast5 files
  output_dir            Output directory

optional arguments:
  -h, --help            show this help message and exit
  --prefix PREFIX       Filename prefix of basecaller output (default: called)
  --gpu GPU             GPU device id (default: 0)
  --read-id-path READ_ID_PATH
                        Basecalling solely on the reads listed in file, with
                        one ID per line (default: None)
  --chunksize CHUNKSIZE
                        Length of signal chunk (default: 20000)
  --batch-size BATCH_SIZE
                        Larger batch size will use more GPU memory (default:4)
  --model MODEL         Path of model checkpoint file (default:
                        model/weights.pth.tar)
  --version             show program's version number and exit
```

### Example for testing basecaller
- Run Coral with the default options to call the Fast5 test data, then align reads to the transcriptome reference 
and calculate the read accuracy. This should be complete within a few minutes.
```shell
cd Coral
python coral.py --prefix coral --gpu 0 test/fast5 test/output 
minimap2 --secondary=no -ax map-ont -t 32 --eqx test/ref/ref.fa test/output/coral.fasta > test/output/coral.sam 
python accuracy.py --samfile test/output/coral.sam
```

- Expected output from test data is as follows: 
```text 
Processing sample: test/output/coral.sam
accuracy  (median/mean): 96.17% / 96.29%
mismatch  (median/mean): 0.86% / 0.97%
insertion (median/mean): 0.94% / 0.97%
deletion  (median/mean): 1.70% / 1.77%
```

### Train from scratch
- Before reproducing our model training, download the [training hdf5](https://zenodo.org/records/4556951/files/rna-train.hdf5?download=1) 
and [validation hdf5](https://zenodo.org/records/4556951/files/rna-valid.hdf5?download=1) created by [RODAN](https://github.com/biodlab/RODAN) 
and place them in the `DATASET` directory. You can also create your own HDF5 dataset, ensuring that the data structure
follows the format defined in the [`dataset.py`](./dataset.py). 

- The opinions for the training script are provided below:
```text
usage: train.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--eval-batch-size EVAL_BATCH_SIZE] [--lr LR] [--limit LIMIT]
                [--valid-limit VALID_LIMIT] [--alphabet ALPHABET]
                [--print-freq PRINT_FREQ] [--eval-freq EVAL_FREQ]
                [--seed SEED]
                DATASET OUTPUT

Training script for Coral v1.0

positional arguments:
  DATASET               Training dataset directory containing rna-train.hdf5
                        and rna-valid.hdf5
  OUTPUT                Output directory (save log and model weights)

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Epoch number (default: 18)
  --batch-size BATCH_SIZE
                        Batch size in training mode (default: 30)
  --eval-batch-size EVAL_BATCH_SIZE
                        Batch size in evaluate mode (default: 30)
  --lr LR               Initial learning rate (default: 0.002)
  --limit LIMIT         Reads number limit in training (default: None)
  --valid-limit VALID_LIMIT
                        Reads number limit in validation (default: 30000)
  --alphabet ALPHABET   Canonical base alphabet (default: NACGT)
  --print-freq PRINT_FREQ
                        Logging step frequency (default: 1)
  --eval-freq EVAL_FREQ
                        Evaluation epoch frequency (default: 1)
  --seed SEED           Random seed for deterministic training (default: 40)

```

- You can start training using our default opinions,
```shell
python train.py DATASET_DIR TRAIN_OUTPUT_DIR
```
and then visualize the loss curve in browser using the `tensorboard` command.
```shell 
tensorboard --logdir TRAIN_OUTPUT_DIR/log --port 8080 
```

### Copyright
Copyright 2024 Zexuan Zhu <zhuzx@szu.edu.cn>
This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
