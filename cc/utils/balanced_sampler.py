"""
Balanced multi-dataset sampler.

We used this strategy since one dataset can dominate the other due to different number of images. 
For instance, when training on multiple datasets of different sizes
(e.g., NUS ~1700 images + Gehler 568 images), a naive concatenation
means the larger dataset dominates training.

So we wanted to ensure that each dataset contributes equally per epoch
by oversampling smaller datasets to match the largest one.
"""

import random
import math
import bisect
from typing import List, Iterator
import torch
from torch.utils.data import Sampler, Dataset


class BalancedMultiDatasetSampler(Sampler):
    """
    Sampler that balances multiple datasets by oversampling smaller ones.
    Each epoch, every dataset contributes the same number of samples.
    """
    def __init__(
        self,
        datasets: List[Dataset],
        dataset_indices: List[List[int]],
        epoch_size: int = None,
        seed: int = 1337,
        shuffle: bool = True
    ):
        self.datasets = datasets
        self.dataset_indices = dataset_indices
        self.n_datasets = len(datasets)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        self.dataset_sizes = [len(indices) for indices in dataset_indices]
        self.max_size = max(self.dataset_sizes)
        
        # This ensures each dataset is seen equally
        if epoch_size is None:
            self.epoch_size = self.max_size * self.n_datasets
        else:
            self.epoch_size = epoch_size
        
        # Each dataset gets equal share of the epoch
        self.samples_per_dataset = self.epoch_size // self.n_datasets
        
        if self.samples_per_dataset == 0:
            raise ValueError(
                f"epoch_size ({self.epoch_size}) is too small for {self.n_datasets} datasets. "
                f"Minimum required: {self.n_datasets}"
            )
        
        print(f"[BalancedMultiDatasetSampler] Initialized:")
        for i, size in enumerate(self.dataset_sizes):
            oversample_ratio = self.samples_per_dataset / size
            print(f"  Dataset {i}: {size} samples, "
                  f"will sample {self.samples_per_dataset} samples/epoch "
                  f"(oversample ratio: {oversample_ratio:.2f}x)")
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch with balanced sampling."""
        
        # Different seed each epoch for different shuffling
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        all_indices = []
        
        for dataset_idx, indices in enumerate(self.dataset_indices):
            dataset_size = len(indices)
            
            if self.shuffle:
                perm = torch.randperm(dataset_size, generator=g).tolist()
                shuffled_indices = [indices[i] for i in perm]
            else:
                shuffled_indices = indices
            
            if dataset_size >= self.samples_per_dataset:
                # Large dataset: just take first N (already shuffled)
                sampled = shuffled_indices[:self.samples_per_dataset]
            else:
                # Small dataset: repeat to fill quota
                # e.g., 200 images, need 1000 → repeat 5 times
                n_repeats = self.samples_per_dataset // dataset_size
                n_remainder = self.samples_per_dataset % dataset_size
                sampled = shuffled_indices * n_repeats + shuffled_indices[:n_remainder]
            
            all_indices.extend(sampled)
        
        # Shuffle everything together so datasets are interleaved in batches
        if self.shuffle:
            random.Random(self.seed + self.epoch + 1).shuffle(all_indices)
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        return self.epoch_size
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch


class ConcatDatasetWithIndices:
    """
    Concatenates multiple datasets while tracking which indices belong to which dataset.
    Like ConcatDataset but also provides get_dataset_indices()
    so the sampler knows which range of indices corresponds to which dataset.
    """
    
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cumulative_sizes = self._compute_cumulative_sizes()
        
    def _compute_cumulative_sizes(self):
        # [0, len(ds0), len(ds0)+len(ds1), ...]
        cumulative = [0]
        for dataset in self.datasets:
            cumulative.append(cumulative[-1] + len(dataset))
        return cumulative
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        
        # Find which dataset this index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        sample_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]
    
    def get_dataset_indices(self) -> List[List[int]]:
        """
        Get list of indices for each dataset in the concatenated dataset.
        """
        indices = []
        for i in range(len(self.datasets)):
            start = self.cumulative_sizes[i]
            end = self.cumulative_sizes[i + 1]
            indices.append(list(range(start, end)))
        return indices


def create_balanced_dataloader(
    datasets: List[Dataset],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 1337,
    epoch_size: int = None,
    collate_fn = None
):
    """
    Create a DataLoader with balanced sampling across multiple datasets.
    """
    from torch.utils.data import DataLoader
    
    concat_dataset = ConcatDatasetWithIndices(datasets)
    dataset_indices = concat_dataset.get_dataset_indices()
    
    sampler = BalancedMultiDatasetSampler(
        datasets=datasets,
        dataset_indices=dataset_indices,
        epoch_size=epoch_size,
        seed=seed,
        shuffle=True
    )
    
    loader = DataLoader(
        concat_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        drop_last=True
    )
    
    return loader, sampler
