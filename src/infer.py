import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .utils import load_yaml, ensure_dir, load_checkpoint, save_tensor_image
from .dataset_vitonhd import make_dataloader
from .models import STVTON


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_dir", type=str, default="outputs/inference")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = make_dataloader(cfg, split=args.split)

    model = STVTON(
        base_channels=cfg["model"].get("base_channels", 64),
        text_dim=cfg["model"].get("text_dim", 512),
        use_text_condition=cfg["model"].get("use_text_condition", True),
    ).to(device)

    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    ensure_dir(args.save_dir)
    ensure_dir(Path(args.save_dir) / "warped")
    ensure_dir(Path(args.save_dir) / "mask")

    for batch in tqdm(loader, desc="Inference"):
        person = batch["person"].to(device)
        cloth = batch["cloth"].to(device)
        mask = batch["mask"].to(device)
        captions = batch["caption"]

        outputs = model(person, cloth, mask, captions)

        name = Path(batch["name"][0]).stem
        save_tensor_image(outputs["tryon"][0], Path(args.save_dir) / f"{name}.png")
        save_tensor_image(outputs["warped_cloth"][0], Path(args.save_dir) / "warped" / f"{name}.png")
        save_tensor_image(outputs["refined_mask"][0].repeat(3,1,1), Path(args.save_dir) / "mask" / f"{name}.png")

    print(f"Saved results to {args.save_dir}")


if __name__ == "__main__":
    main()
