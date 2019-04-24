import torch
import numpy as np
import torch.utils.data as data


class Subset(data.Dataset):
    def __init__(self, dataset, indices=None):
        """
        Subset of dataset given by indices.
        """
        super(Subset, self).__init__()
        self.dataset = dataset
        self.indices = indices

        if self.indices is None:
            self.n_samples = len(self.dataset)
        else:
            self.n_samples = len(self.indices)
            assert self.n_samples >= 0 and \
                self.n_samples <= len(self.dataset), \
                "length of {} incompatible with dataset of size {}"\
                .format(self.n_samples, len(self.dataset))

    def __getitem__(self, idx):
        if torch.is_tensor(idx) and idx.dim():
            res = [self[iidx] for iidx in idx]
            return torch.stack([x[0] for x in res]), torch.LongTensor([x[1] for x in res])
        if self.indices is None:
            return self.dataset[idx]
        else:
            return self.dataset[self.indices[idx]]

    def __len__(self):
        return self.n_samples


def random_subsets(subset_sizes, n_total, seed=None, replace=False):
    """
    Return subsets of indices, with sizes given by the iterable
    subset_sizes, drawn from {0, ..., n_total - 1}
    Subsets may be distinct or not according to the replace option.
    Optional seed for deterministic draw.
    """
    # save current random state
    state = np.random.get_state()
    sum_sizes = sum(subset_sizes)
    assert sum_sizes <= n_total

    np.random.seed(seed)

    total_subset = np.random.choice(n_total, size=sum_sizes,
                                    replace=replace)
    perm = np.random.permutation(total_subset)
    res = []
    start = 0
    for size in subset_sizes:
        res.append(perm[start: start + size])
        start += size
    # restore initial random state
    np.random.set_state(state)
    return res
