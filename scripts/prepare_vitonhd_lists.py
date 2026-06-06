from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=None)
    parser.add_argument("--caption", default="a photo of a person wearing the target garment")
    args = parser.parse_args()

    root = Path(args.root)
    image_dir = root / args.split / "image"
    cloth_dir = root / args.split / "cloth"

    images = sorted([p.name for p in image_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    cloths = sorted([p.name for p in cloth_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    n = min(len(images), len(cloths))
    out = Path(args.out) if args.out else root / f"{args.split}_pairs.txt"

    with open(out, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(f"{images[i]} {cloths[i]} {args.caption}\n")

    print(f"Wrote {n} pairs to {out}")


if __name__ == "__main__":
    main()
