import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler

import random
import numpy as np

from utils import FilterByBR
from data.dataset import TiledDataset
from data.transforms import build_transforms

class BuildingDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for building segmentation datasets.
    
        Required args in data config:
        - dataset: str, one of ["whu", "massachusetts", "inria"]
        - patch_size: int, size of the square patches to extract from the images
        - train_batch_size: int
        - val_batch_size: int
        Optional args in data config:
        - normalization: str, one of ["imagenet", "standard"], default "imagenet"
        - building_threshold: float, default 0.05, minimum ratio of building pixels in a patch to be included in training
        - train_transform: dict, keys are transform names and values are probabilities.
        - val_transform: dict, keys are transform names and values are probabilities.
        - samples_per_epoch: int, if set, limits the number of samples in each training epoch (only for training dataloader)
        - num_workers: int, number of workers for data loading. If not set, defaults to 4 or 2 if platform is colab.
    """
    def __init__(self, config, train_h5, val_h5):
        super().__init__()
        self.config = config
        self.train_h5 = train_h5
        self.val_h5 = val_h5
        self.data_cfg = config["data"]

        self.num_workers = self.config.get("num_workers", self.get_num_workers(config))

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = self.build_dataset()

    def train_dataloader(self):
        
        generator = torch.Generator()
        generator.manual_seed(self.config["seed"])

        samples_per_epoch = self.data_cfg.get("samples_per_epoch", None)
        if samples_per_epoch is not None and len(self.train_dataset) > samples_per_epoch:
            sampler = RandomSampler(
                self.train_dataset,
                replacement=False,
                num_samples=samples_per_epoch,
                generator=generator
            )
            shuffle = False

            # print(list(iter(sampler))[:20])
        elif samples_per_epoch is not None and len(self.train_dataset) < samples_per_epoch:
            raise ValueError(f"Samples per epoch {samples_per_epoch} is greater than the dataset size {len(self.train_dataset)}, Case not handled.")
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.data_cfg["train_batch_size"],
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            sampler=sampler,
            generator=generator,
            worker_init_fn=self.worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_cfg["val_batch_size"],
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            worker_init_fn=self.worker_init_fn
        )

    def build_dataset(self,):
        dataset_name = self.data_cfg["dataset"]

        if dataset_name not in ["whu", "massachusetts", "inria"]:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Available datasets: [whu, massachusetts, inria]")

        train_transform = build_transforms(self.data_cfg, mode="train", seed=self.config["seed"])
        val_transform = build_transforms(self.data_cfg, mode="val", seed=self.config["seed"])

        train_dataset = TiledDataset(h5_path=self.train_h5,
                                    transform=train_transform,
                                    patch_size=self.data_cfg["patch_size"],
                                    filter_func=FilterByBR(self.data_cfg.get("building_threshold", 0.05)),
                                    use_cache=True
                                    )

        val_dataset = TiledDataset(h5_path=self.val_h5,
                                    transform=val_transform,
                                    patch_size=self.data_cfg["patch_size"],
                                    use_cache=True
                                    )


        return train_dataset, val_dataset
    
    def worker_init_fn(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_num_workers(self, config):
        if config.get("platform", None) == "colab":
            return 2
        return 4
