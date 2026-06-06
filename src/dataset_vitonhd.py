from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .utils import pil_loader, mask_loader, find_file


class VITONHDDataset(Dataset):
    """
    Dataset loader linh hoạt cho VITON-HD style.

    Output:
        person: ảnh người gốc, tensor [-1,1]
        cloth: ảnh áo, tensor [-1,1]
        target: ảnh ground-truth try-on, tensor [-1,1]
        mask: mask vùng áo/parse, tensor [0,1]
        caption: text prompt
        name: tên ảnh person
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        pairs_file: str = "train_pairs.txt",
        image_size=(256, 192),
        use_parse_mask: bool = True,
        default_caption: str = "a photo of a person wearing the target garment",
    ):
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.image_size = tuple(image_size)
        self.use_parse_mask = use_parse_mask
        self.default_caption = default_caption

        pair_path = self.root / pairs_file
        if not pair_path.exists():
            raise FileNotFoundError(f"Không tìm thấy pairs file: {pair_path}")

        self.samples = self._read_pairs(pair_path)

        self.img_tf = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _read_pairs(self, pair_path: Path) -> List[Tuple[str, str, str]]:
        samples = []
        with open(pair_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name = parts[0]
                cloth_name = parts[1] if len(parts) > 1 else parts[0]
                caption = " ".join(parts[2:]) if len(parts) > 2 else self.default_caption
                samples.append((person_name, cloth_name, caption))
        return samples

    def _make_cloth_mask(self, cloth: Image.Image):
        gray = cloth.convert("L")
        import numpy as np
        arr = np.array(gray)
        mask = (arr < 245).astype("uint8") * 255
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        person_name, cloth_name, caption = self.samples[idx]

        person_path = find_file(self.split_dir / "image", person_name)
        cloth_path = find_file(self.split_dir / "cloth", cloth_name)

        person = pil_loader(person_path)
        cloth = pil_loader(cloth_path)
        target = person.copy()

        mask_path = find_file(self.split_dir / "image-parse-v3", person_name)
        if self.use_parse_mask and mask_path.exists():
            mask = mask_loader(mask_path)
            mask = mask.point(lambda p: 255 if p > 0 else 0)
        else:
            mask = self._make_cloth_mask(cloth)

        return {
            "person": self.img_tf(person),
            "cloth": self.img_tf(cloth),
            "target": self.img_tf(target),
            "mask": self.mask_tf(mask).clamp(0, 1),
            "caption": caption,
            "name": person_name,
            "cloth_name": cloth_name,
        }


def make_dataloader(cfg, split="train"):
    from torch.utils.data import DataLoader

    data_cfg = cfg["data"]
    pairs_file = data_cfg["train_pairs"] if split == "train" else data_cfg["test_pairs"]

    ds = VITONHDDataset(
        root=data_cfg["root"],
        split=split,
        pairs_file=pairs_file,
        image_size=data_cfg["image_size"],
        use_parse_mask=data_cfg.get("use_parse_mask", True),
        default_caption=data_cfg.get("default_caption", ""),
    )

    train_cfg = cfg.get("train", {})
    return DataLoader(
        ds,
        batch_size=train_cfg.get("batch_size", 4) if split == "train" else 1,
        shuffle=(split == "train"),
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=True,
    )
