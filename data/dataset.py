import h5py
import numpy as np
from torch.utils.data import Dataset

class TiledDataset(Dataset):
    def __init__(self, h5_path, patch_size, stride=None, transform=None, filter_func=None, use_cache=False, indices=None):
        self.h5_path = h5_path
        self.patch_size = patch_size
        self.stride = stride if stride else patch_size//2
        self.transform = transform
        self.file = None
        self.filter_func = filter_func

        # build index of all patches upfront
        self.patch_index = []  # (image_idx, y, x)

        with h5py.File(h5_path, 'r') as f:
            self.n_images = len(f['images'])
            h, w = f['images'].shape[1], f['images'].shape[2]
        
        self.indices = set(indices) if indices is not None else range(self.n_images)

        self.image_h = h
        self.image_w = w

        self.pad_h = (self.stride - h % self.stride) % self.stride
        self.pad_w = (self.stride - w % self.stride) % self.stride

        self._build_index()

        if filter_func:
            self._filter_patch_index()

        self.use_cache = use_cache

        self.cached_img_idx = None
        self.cached_image = None
        self.cached_mask = None

    def _build_index(self):
        p = self.patch_size
        s = self.stride
        h, w = self.image_h, self.image_w

        padded_h = h + self.pad_h
        padded_w = w + self.pad_w

        for img_idx in range(self.n_images):
            if img_idx not in self.indices:
                continue
            for y in range(0, padded_h - p + 1, s):
                for x in range(0, padded_w - p + 1, s):
                    self.patch_index.append((img_idx, y, x))

    def _pad_image(self, image):
        # image: H x W x 3 numpy
        if self.pad_h > 0 or self.pad_w > 0:
            image = np.pad(image, ((0, self.pad_h), (0, self.pad_w), (0, 0)), mode='reflect')
        return image

    def _pad_mask(self, mask):
        # mask: H x W numpy
        if self.pad_h > 0 or self.pad_w > 0:
            mask = np.pad(mask, ((0, self.pad_h), (0, self.pad_w)), mode='reflect')
        return mask

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        img_idx, y, x = self.patch_index[idx]
        p = self.patch_size

        if self.use_cache:
            if img_idx != self.cached_img_idx:
                self.cached_img_idx = img_idx

                image = self.file['images'][img_idx]
                mask = self.file['masks'][img_idx]

                self.cached_image = self._pad_image(image)
                self.cached_mask = self._pad_mask(mask)

            image = self.cached_image
            mask = self.cached_mask
        else:
            image = self.file['images'][img_idx]
            mask = self.file['masks'][img_idx]

            image = self._pad_image(image)
            mask = self._pad_mask(mask)

        image_patch = image[y:y+p, x:x+p]
        mask_patch = mask[y:y+p, x:x+p]

        if self.transform:
            transformed = self.transform(image=image_patch, mask=mask_patch)
            image_patch = transformed['image']
            mask_patch = transformed['mask']

        if isinstance(mask_patch, np.ndarray):
            # Triggers if no transform applied
            mask_patch = (mask_patch > 0).astype(np.float32)
        else:
            mask_patch = (mask_patch > 0).float().unsqueeze(0)  # Add channel dimension for PyTorch

        return image_patch, mask_patch, img_idx, y, x

    def _filter_patch_index(self):
        filtered = []

        with h5py.File(self.h5_path, 'r') as f:
            masks = f["masks"]

            current_img = None

            for img_idx, y, x in self.patch_index:

                if img_idx != current_img:
                    current_img = img_idx
                    mask = self._pad_mask(masks[img_idx])

                patch = mask[y:y+self.patch_size, x:x+self.patch_size]

                if self.filter_func(patch):
                    filtered.append((img_idx, y, x))

            self.patch_index = filtered
