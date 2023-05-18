import numpy as np
from utils.utils import save_pickle, load_pickle


def sort_by_class(s):
    sy = np.sort(s[1][1])
    indices = np.argsort(s[1][1])
    return s[0], (s[1][0][indices], sy)


# continual sequence generator
class SequenceGenerator:
    def __init__(self, file_path="./data/database.pk"):
        self.dict = load_pickle(file_path)

    def gen_sequence(
        self,
        seq_len=1000,
        dataset_list=["mnist", "cifar10", "svhn"],
        correlation="ci",
        fold="test",
        shuffle_class=True,
        shuffle_dset=False,
    ):

        seq_chunks_x = []
        seq_chunks_y = []

        dset_len = seq_len // len(dataset_list)

        for i, dset in enumerate(dataset_list):
            dset_chunks_x = []
            dset_chunks_y = []

            cur_dset_len = dset_len
            if i == len(dataset_list) - 1:
                cur_dset_len = dset_len + seq_len % dset_len

            class_len = cur_dset_len // len(self.dict[dset][fold].keys())
            for j, y in enumerate(self.dict[dset][fold].keys()):
                chunk_len = class_len
                if j == len(self.dict[dset][fold].keys()) - 1:
                    chunk_len = class_len + cur_dset_len % class_len

                permutation = np.random.permutation(self.dict[dset][fold][y].shape[0])
                chunk = self.dict[dset][fold][y][permutation][:chunk_len]
                dset_chunks_x.append(chunk)
                dset_chunks_y.append((y * np.ones(chunk_len)))

            seq_chunks_x += dset_chunks_x
            seq_chunks_y += dset_chunks_y

        seq_x = np.concatenate(seq_chunks_x, 0)
        seq_y = np.concatenate(seq_chunks_y, 0)

        if correlation == "iid":
            permutation = np.random.permutation(seq_x.shape[0])
            seq_x, seq_y = seq_x[permutation], seq_y[permutation]

        return seq_x, seq_y