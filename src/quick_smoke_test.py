import torch
from .models import STVTON
from .losses import STVTONLoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = STVTON(base_channels=16).to(device)
    criterion = STVTONLoss()

    b, h, w = 2, 128, 96
    batch = {
        "person": torch.randn(b, 3, h, w).to(device),
        "cloth": torch.randn(b, 3, h, w).to(device),
        "mask": torch.rand(b, 1, h, w).to(device),
        "target": torch.randn(b, 3, h, w).to(device),
        "caption": ["red shirt", "blue jacket"],
    }

    out = model(batch["person"], batch["cloth"], batch["mask"], batch["caption"])
    loss, logs = criterion(out, batch)
    loss.backward()

    print("Smoke test OK")
    print(logs)
    for k, v in out.items():
        print(k, tuple(v.shape))


if __name__ == "__main__":
    main()
