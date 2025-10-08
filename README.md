# Coral: a dual context-aware basecaller for nanopore direct RNA sequencing

-----

## 🚀 Download and Install

### System Dependencies

  * NVIDIA GPU with CUDA compute capability \>= 8.x (e.g., Ampere, Ada, or Hopper GPUs like A100, RTX 3090, RTX 4090, H100)
  * NVIDIA driver version \>= 450.80.02
  * CUDA Toolkit \>= 11.8

Coral can be installed on Linux and has been tested on Ubuntu 22.04 with an RTX 3090 GPU.

### Install from Docker 

We recommend installing Coral using the pre-built Docker image. Ensure you have Docker and the NVIDIA Container Toolkit installed by 
following this [tutorial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```shell 
docker pull chobits323/coral:latest
```

### Install from conda and pip 

Alternatively, you can install Coral in a [conda](https://www.anaconda.com/download/success) environment. 
This installation usually takes ~30 minutes.

```shell
# create conda environment 
conda create -n coral -c conda-forge python==3.10.16
conda activate coral

# install minimap2 
conda install -c bioconda -c conda-forge minimap2==2.28

# install pytorch and torchtune 
pip install torch==2.6.0 torchao==0.8.0 --index-url https://download.pytorch.org/whl/cu118  
pip install torchtune==0.5.0

# install flash-attn, this may take a long time for compiling 
pip install packaging==24.2 ninja==1.11.1.3 psutil==6.1.1 einops==0.8.0 
pip install flash-attn==2.7.4.post1 --no-build-isolation

# install coral package using pip
pip install coral-call 
```

-----

## ⚙️ Usage

### Basic

#### Using Docker
```bash
docker run --rm -it --gpus=all --ipc=host chobits323/coral:latest coral [-h] {download,basecall,train,accuracy} ...
```

#### Using Conda Environment

```bash 
conda activate coral
python -m coral [-h] {download,basecall,train,accuracy} ... 
```

#### Subcommands

```text 
usage: python -m coral [-h] [--version] {download,basecall,train,accuracy} ...

Nanopore Direct-RNA sequencing basecaller

positional arguments:
  {download,basecall,train,accuracy}
    download            Download the pretrained model checkpoints
    basecall            Basecalling
    train               Train the model
    accuracy            Compute the read accuracy given a SAM file

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

### 📥 Download Pretrained Models

#### Download Options

**Note:** You don't need to do this if you're using the Docker pre-built image, as it already includes the models.

```text 
usage: python -m coral download [-h] [--list] [--all] [--model {RNA002,RNA002_FAST,RNA004,RNA004_FAST}]

options:
  -h, --help            show this help message and exit
  --list                List available models
  --all                 Download all models
  --model {RNA002,RNA002_FAST,RNA004,RNA004_FAST}
                        Model to download
```

#### List of available models

| Model        | URL                                                                 | MD5                                |
|--------------|---------------------------------------------------------------------|-------------------------------------|
| RNA002       | https://zenodo.org/records/15590113/files/rna002_model.pth         | 6e41a6ba3e5cb0f5af10675c1ed3b9c6    |
| RNA002_FAST  | https://zenodo.org/records/15590113/files/rna002_fast_model.pth    | 480da3f5bb9dc7a2286e9c2844b7782a    |
| RNA004       | https://zenodo.org/records/15590113/files/rna004_model.pth         | ab09425e6e40bf0d35e4a00f1b5ec383    |
| RNA004_FAST  | https://zenodo.org/records/15590113/files/rna004_fast_model.pth    | c86f8e832c0acf0bfa01a32af25717ba    |


### Basecalling

#### Basecalling options

```text
usage: python -m coral basecall [-h] --input INPUT --output OUTPUT --kit {RNA002,RNA004} [--fast] [--gpu GPU] [--gpus GPUS] [--batch-size BATCH_SIZE] [--beam-size BEAM_SIZE] [--prefix PREFIX] [--seed SEED] [--no-deterministic]
                                [--parse-fast5-meta] [--reads-file READS_FILE] [--keep-split-reads]

options:
  -h, --help            show this help message and exit
  --input INPUT         Directory containing fast5 files or Single pod5 file (default: None)
  --output OUTPUT       Output directory (default: None)
  --kit {RNA002,RNA004}
                        RNA002 or RNA004 sequencing kit (default: None)
  --fast                Use FAST mode that outputs k consecutive bases per step (default: False)
  --gpu GPU             GPU device id (default: 0)
  --gpus GPUS           Comma-separated GPU device ids for multi-gpu basecalling, e.g. 0,1,2 (default: None)
  --batch-size BATCH_SIZE
                        Larger batch size will use more GPU memory (default: 500)
  --beam-size BEAM_SIZE
                        Beam size (default: None)
  --prefix PREFIX       Filename prefix of basecaller output (default: coral)
  --seed SEED           Seed for random number generators (default: 40)
  --no-deterministic    Disable CUDNN deterministic algorithm (default: False)
  --parse-fast5-meta    Parse multi-fast5 meta data (default: False)
  --reads-file READS_FILE
                        Basecalling solely on the reads listed in file, with one ID per line (default: None)
  --keep-split-reads    Keep temporary split read files (default: False)
```

#### Basecalling Examples

1.  **RNA002 Example (Fast5 data, Docker):** Use Docker to run Coral on the RNA002 Fast5 example data. 
Then, align the reads to the transcriptome reference and compute the accuracy. This should be complete within a few minutes.

     ```shell
     docker run --rm -it --gpus=all --ipc=host -v ./example:/data chobits323/coral:latest coral basecall \
                    --input /data/rna002/fast5 \
                    --output /data/rna002/output \
                    --kit RNA002 \
                    --prefix coral  
   
     docker run --rm -it -v ./example:/data chobits323/coral:latest minimap2 \
            --secondary=no -ax map-ont -t 32 --eqx /data/rna002/ref/ref.fa /data/rna002/output/coral.fasta -o /data/rna002/output/coral.sam
     
     docker run --rm -it --gpus=all --ipc=host -v ./example:/data chobits323/coral:latest coral accuracy --samfile /data/rna002/output/coral.sam 
     ``` 
    
     The expected output for the RNA002 example is:

     ```text
     Processing sample: /data/rna002/output/coral.sam
     accuracy  (median/mean): 97.43% / 96.55%
     mismatch  (median/mean): 0.60% / 0.99%
     insertion (median/mean): 0.62% / 0.89%
     deletion  (median/mean): 1.06% / 1.57%
     read length  (median/mean): 863 / 914
     ```

2.  **RNA004 Example (POD5 data, Conda):** In the conda environment, run Coral on the RNA004 POD5 test data using the FAST model. 
Afterward, align the reads and calculate the accuracy. This should be complete within a few minutes.

    ```shell
    python -m coral basecall --input example/rna004/pod5/example.pod5 --output example/rna004/output --kit RNA004 --fast --beam-size 1 --prefix coral
    minimap2 --secondary=no -ax lr:hq -t 32 --eqx example/rna004/ref/ref.fa example/rna004/output/coral.fasta -o example/rna004/output/coral.sam
    python -m coral accuracy --samfile example/rna004/output/coral.sam    
    ```
    
    The expected output for the RNA004 example is:
    
    ```text
    Processing sample: example/rna004/output/coral.sam
    The number of supplementary alignments in this sample is  2
    The number of unmapped reads in this sample is  1
    accuracy  (median/mean): 99.61% / 98.24%
    mismatch  (median/mean): 0.07% / 0.40%
    insertion (median/mean): 0.00% / 0.29%
    deletion  (median/mean): 0.05% / 1.07%
    read length  (median/mean): 727 / 931
    ```

### 🧠 Training

#### Dataset

- RNA002 training dataset can be downloaded from [training hdf5](https://emailszueducn-my.sharepoint.com/:u:/g/personal/2060271006_email_szu_edu_cn/Ee3O8i9SQflEtB8tH-JR8zUBFV5i5GK8kpvY0Keb9WA0ZA?e=0YzjZY)
and [validation hdf5](https://emailszueducn-my.sharepoint.com/:u:/g/personal/2060271006_email_szu_edu_cn/EZ1RJWsmnhpKm4G8iZQNLO4BpHGmHqvYxK-e-3hovT9Mgw?e=VciWpU). 
- RNA004 training dataset (partial, ~2.16 million chunks) can be downloaded from [training hdf5](https://emailszueducn-my.sharepoint.com/:u:/g/personal/2060271006_email_szu_edu_cn/Ecj7dKDoopZCkpEOms8_PqoBlbx2ANHo7i0TyUCDVZ1inw?e=PkG8W8)
and [validation hdf5](https://emailszueducn-my.sharepoint.com/:u:/g/personal/2060271006_email_szu_edu_cn/EZPGFFZr-eRHh44jxMqqD-EBiLAN9Z_RobEStPe-uVyACg?e=k7NN4J). 
- Place these HDF5 files in the `DATASET` directory. You can also create your own HDF5 dataset, ensuring that the data structure
follows the format defined in the [`dataset.py`](./coral/dataset.py). 

#### Training options

```text
usage: python -m coral train [-h] --data DATA --output OUTPUT [--dist-url DIST_URL] [--ngpus-per-node NGPUS_PER_NODE] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR] [--k K]
                             [--decoder-layers DECODER_LAYERS] [--monotonic-warmup-steps MONOTONIC_WARMUP_STEPS] [--pretrained-checkpoint PRETRAINED_CHECKPOINT] [--seed SEED]

options:
  -h, --help            show this help message and exit
  --data DATA           Training dataset directory containing rna-train.hdf5
  --output OUTPUT       Output directory (save log and model weights)
  --dist-url DIST_URL   URL specifying how to initialize the process group in multi-gpu training (default: tcp://127.0.0.1:23456)
  --ngpus-per-node NGPUS_PER_NODE
                        Number of GPUs used for training (default: 1)
  --epochs EPOCHS       Number of training epochs (default: 10)
  --batch-size BATCH_SIZE
                        Training batch size (default: 128)
  --lr LR               Initial learning rate (default: 0.0002)
  --k K                 Symbol prediction granularity, k > 1 for multi-base prediction per auto-regressive step (default: 1)
  --decoder-layers DECODER_LAYERS
                        Number of decoder layers (default: 12)
  --monotonic-warmup-steps MONOTONIC_WARMUP_STEPS
                        Warmup steps without adding monotonic regularization loss at the training start (default: None)
  --pretrained-checkpoint PRETRAINED_CHECKPOINT
                        Pretrained model checkpoint (default: None)
  --seed SEED           Random seed for deterministic training (default: 40)
```

#### Training Examples

1.  **Initial Training (k=1):** First, train the model to predict single base per step (k=1) using only cross-entropy loss. 
You can visualize the loss curve with `tensorboard`:
    ```shell
    python -m coral train --data DATASET_DIR --output TRAIN_OUTPUT_DIR --k 1 --decoder-layers 12 
    tensorboard --logdir TRAIN_OUTPUT_DIR/log --port 8080 
    ```
2.  **Add Monotonic Alignment Regularization:** Next, load the best checkpoint from the initial training. 
Continue training, this time adding monotonic alignment regularization loss after thousands of warmup steps.
    ```shell
    python -m coral train --data DATASET_DIR --output TRAIN_OUTPUT_DIR_2 --k 1 --decoder-layers 12 --pretrained-checkpoint TRAIN_OUTPUT_DIR/weights/{best_epoch_checkpoint} --monotonic-warmup-steps 3000
    ```
3.  **Multi-base Prediction (k=5):** Finally, load the best checkpoint from the previous training. 
Train the model to predict *k* (e.g., 5) consecutive bases per step, again including regularization loss after thousands of warmup steps.
    ```shell
    python -m coral train --data DATASET_DIR --output TRAIN_OUTPUT_DIR_3 --k 5 --decoder-layers 12 --pretrained-checkpoint TRAIN_OUTPUT_DIR_2/weights/{best_epoch_checkpoint} --monotonic-warmup-steps 5000
    ```

-----

## ©️ Copyright

Copyright 2025 Zexuan Zhu <zhuzx@szu.edu.cn>.<br>
This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
