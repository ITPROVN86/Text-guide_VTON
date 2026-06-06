import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path, model, optimizer=None, epoch=0, extra=None):
    ensure_dir(Path(path).parent)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def denorm_tensor(x):
    return (x.clamp(-1, 1) + 1.0) / 2.0


def save_tensor_image(tensor, path):
    from torchvision.utils import save_image
    ensure_dir(Path(path).parent)
    save_image(denorm_tensor(tensor.detach().cpu()), path)


def pil_loader(path):
    return Image.open(path).convert("RGB")


def mask_loader(path):
    return Image.open(path).convert("L")


def find_file(folder, name):
    p = Path(folder) / name
    if p.exists():
        return p
    stem = Path(name).stem
    for ext in [".png", ".jpg", ".jpeg"]:
        q = Path(folder) / f"{stem}{ext}"
        if q.exists():
            return q
    return p
