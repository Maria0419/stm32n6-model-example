from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SquareSegDataset(Dataset):
    def __init__(self, root, image_height=128, image_width=128, augment=False):
        self.root = Path(root)
        self.size = (image_width, image_height)
        self.augment = augment
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {self.root}")

        self.samples = []
        missing = []
        for image_path in sorted(self.root.rglob("*_image.tif")):
            mask_path = image_path.with_name(image_path.name.replace("_image.tif", "_label.tif"))
            if mask_path.is_file():
                self.samples.append((image_path, mask_path))
            else:
                missing.append(mask_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("L")
            if image.size != self.size:
                image = image.resize(self.size, Image.Resampling.BILINEAR)
            image = np.asarray(image, dtype=np.float32) / 255.0
        with Image.open(mask_path) as source:
            mask = source.convert("L")
            if mask.size != self.size:
                mask = mask.resize(self.size, Image.Resampling.NEAREST)
            mask = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)

        if self.augment and torch.rand(()).item() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)
