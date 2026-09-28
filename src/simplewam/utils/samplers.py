from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class ResumableWeightedEpochSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Sized,
        weights,
        seed: int,
        batch_size: int,
        num_processes: int,
        num_samples: int | None = None,
        replacement: bool = True,
    ):
        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1D, got shape {tuple(self.weights.shape)}.")
        if len(self.weights) != len(dataset):
            raise ValueError(f"weights length mismatch: {len(self.weights)} vs dataset length {len(dataset)}.")
        if not torch.isfinite(self.weights).all():
            raise ValueError("weights must be finite.")
        if torch.any(self.weights < 0):
            raise ValueError("weights must be non-negative.")
        if float(self.weights.sum().item()) <= 0:
            raise ValueError("weights must contain at least one positive value.")

        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.num_samples = int(num_samples) if num_samples is not None else len(dataset)
        self.replacement = bool(replacement)
        if not self.replacement and self.num_samples > len(dataset):
            raise ValueError("num_samples cannot exceed dataset length when replacement=false.")
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.multinomial(
            self.weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
            generator=g,
        ).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
