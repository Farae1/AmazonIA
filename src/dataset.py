import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio

class GeoTiffDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # MUDANÇA: Lê TODAS as bandas (ex: 4 bandas)
        # O shape será (C, H, W), por exemplo (4, H, W)
        with rasterio.open(self.image_paths[idx]) as src:
            img = src.read().astype(np.float32)

        # Máscara 1-banda
        with rasterio.open(self.mask_paths[idx]) as msk:
            mask = msk.read(1).astype(np.int64)

        # MUDANÇA: Normalização Min-Max por canal
        if img.max() > 1.0: # Evita normalizar se já estiver normalizado
            for c in range(img.shape[0]):
                min_val = img[c].min()
                max_val = img[c].max()
                img[c] = (img[c] - min_val) / (max_val - min_val + 1e-8)

        # Converte máscara para binária
        mask = np.where(mask > 0, 1, 0)

        # A linha "np.expand_dims" foi removida, pois img já é (C, H, W)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)
