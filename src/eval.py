import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .utils import load_yaml, find_file
from .dataset_vitonhd import VITONHDDataset
from .metrics import compute_psnr_ssim, OptionalLPIPS, OptionalCLIPScore, try_compute_fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_csv", type=str, default="outputs/metrics.csv")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_cfg = cfg["data"]
    pairs_file = data_cfg["train_pairs"] if args.split == "train" else data_cfg["test_pairs"]

    ds = VITONHDDataset(
        root=data_cfg["root"],
        split=args.split,
        pairs_file=pairs_file,
        image_size=data_cfg["image_size"],
        use_parse_mask=data_cfg.get("use_parse_mask", True),
        default_caption=data_cfg.get("default_caption", ""),
    )

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(data_cfg["root"]) / args.split / "image"

    lpips_metric = OptionalLPIPS(device)
    clip_metric = OptionalCLIPScore(device)

    rows = []
    for sample in tqdm(ds.samples, desc="Evaluate"):
        person_name, cloth_name, caption = sample
        name = Path(person_name).stem
        pred_path = pred_dir / f"{name}.png"
        gt_path = find_file(gt_dir, person_name)

        if not pred_path.exists() or not gt_path.exists():
            continue

        psnr, ssim = compute_psnr_ssim(pred_path, gt_path)
        lpips = lpips_metric(pred_path, gt_path)
        clip_score = clip_metric(pred_path, caption)

        rows.append({
            "name": person_name,
            "cloth": cloth_name,
            "caption": caption,
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": lpips,
            "CLIPScore": clip_score,
        })

    df = pd.DataFrame(rows)
    Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_csv, index=False)

    summary = {
        "N": len(df),
        "PSNR_mean": df["PSNR"].mean() if "PSNR" in df else None,
        "SSIM_mean": df["SSIM"].mean() if "SSIM" in df else None,
        "LPIPS_mean": df["LPIPS"].dropna().mean() if "LPIPS" in df else None,
        "CLIPScore_mean": df["CLIPScore"].dropna().mean() if "CLIPScore" in df else None,
    }

    fid = try_compute_fid(pred_dir, gt_dir, device=device)
    summary["FID"] = fid

    summary_path = str(Path(args.save_csv).with_suffix(".summary.csv"))
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("Summary:")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Saved per-image metrics to {args.save_csv}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
