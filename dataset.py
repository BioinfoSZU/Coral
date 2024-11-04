import os
import h5py
import numpy as np
from torch.utils.data import Dataset


class RNALabelDataset(Dataset):
    def __init__(self, dataset_dir, read_limit=None, is_validate=False, use_fp32=False, use_shuffle_indices=False, ):
        super(RNALabelDataset, self).__init__()
        if is_validate:
            self.file_path = os.path.join(dataset_dir, 'rna-valid.hdf5')
        else:
            self.file_path = os.path.join(dataset_dir, 'rna-train.hdf5')

        h5 = h5py.File(self.file_path, "r")
        self.read_count = len(h5["events"])
        h5.close()
        if read_limit is not None:
            self.read_count = read_limit
        self.use_fp32 = use_fp32

        self.indices = np.arange(self.read_count)
        if use_shuffle_indices:
            rng = np.random.default_rng(seed=42)
            rng.shuffle(self.indices)

    def __len__(self):
        return self.read_count

    def __getitem__(self, i):
        h5 = h5py.File(self.file_path, "r")
        signal = h5["events"][self.indices[i]]
        seqs = h5["labels"][self.indices[i]]
        seqs[seqs == 0] = 1
        seqs_len = h5["labels_len"][self.indices[i]]

        h5.close()
        if self.use_fp32:
            return (
                signal.astype(np.float32),
                seqs.astype(np.int64),
                seqs_len.astype(np.int64)
            )
        else:
            return (
                signal.astype(np.float16),
                seqs.astype(np.int64),
                seqs_len.astype(np.int64)
            )
