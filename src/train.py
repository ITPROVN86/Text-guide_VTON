import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .utils import load_yaml, set_seed, ensure_dir, save_checkpoint, save_tensor_image
from .dataset_vitonhd import make_dataloader
from .models import STVTON
from .losses import STVTONLoss


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, out_root):
    model.train()
    running = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        person = batch["person"].to(device)
        cloth = batch["cloth"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        captions = batch["caption"]

        batch["target"] = target
        batch["mask"] = mask

        outputs = model(person, cloth, mask, captions)
        loss, logs = criterion(outputs, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += logs["loss_total"]
        pbar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})

        if step == 0:
            sample_dir = Path(out_root) / "samples"
            ensure_dir(sample_dir)
            save_tensor_image(outputs["tryon"][0], sample_dir / f"epoch_{epoch:03d}_tryon.png")
            save_tensor_image(outputs["warped_cloth"][0], sample_dir / f"epoch_{epoch:03d}_warped.png")
            save_tensor_image(outputs["refined_mask"][0].repeat(3,1,1), sample_dir / f"epoch_{epoch:03d}_mask.png")

    return running / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ensure_dir(cfg["output"]["checkpoint_dir"])
    ensure_dir(cfg["output"]["log_dir"])

    train_loader = make_dataloader(cfg, split="train")

    model = STVTON(
        base_channels=cfg["model"].get("base_channels", 64),
        text_dim=cfg["model"].get("text_dim", 512),
        use_text_condition=cfg["model"].get("use_text_condition", True),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"].get("lr", 2e-4),
        betas=(cfg["train"].get("beta1", 0.5), 0.999),
    )

    criterion = STVTONLoss(
        lambda_l1=cfg["train"].get("lambda_l1", 10.0),
        lambda_perceptual=cfg["train"].get("lambda_perceptual", 1.0),
        lambda_mask=cfg["train"].get("lambda_mask", 2.0),
    )

    best_loss = float("inf")
    epochs = cfg["train"].get("epochs", 20)

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg["output"]["root"]
        )

        print(f"[Epoch {epoch}] avg_loss={avg_loss:.6f}")

        latest_path = Path(cfg["output"]["checkpoint_dir"]) / "st_vton_latest.pth"
        save_checkpoint(latest_path, model, optimizer, epoch, {"avg_loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = Path(cfg["output"]["checkpoint_dir"]) / "st_vton_best.pth"
            save_checkpoint(best_path, model, optimizer, epoch, {"avg_loss": avg_loss})

    print("Training finished.")


if __name__ == "__main__":
    main()
