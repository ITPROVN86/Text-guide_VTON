from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_rgb01(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def compute_psnr_ssim(pred_path, gt_path):
    pred = load_rgb01(pred_path)
    gt = load_rgb01(gt_path, size=(pred.shape[1], pred.shape[0]))

    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


class OptionalLPIPS:
    def __init__(self, device):
        self.device = device
        self.model = None
        try:
            import lpips
            self.model = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as e:
            print(f"[Warning] LPIPS disabled: {e}")

    def __call__(self, pred_path, gt_path):
        if self.model is None:
            return None
        import torchvision.transforms as T
        img_tf = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        pred = Image.open(pred_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB").resize(pred.size)
        pred = img_tf(pred).unsqueeze(0).to(self.device)
        gt = img_tf(gt).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.model(pred, gt).item())


class OptionalCLIPScore:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        try:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self.model = self.model.to(device).eval()
        except Exception as e:
            print(f"[Warning] CLIPScore disabled: {e}")

    def __call__(self, img_path, text):
        if self.model is None:
            return None
        img = Image.open(img_path).convert("RGB")
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            score = (image_features @ text_features.T).item()
        return float(score)


def try_compute_fid(pred_dir, gt_dir, device="cuda"):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        import torchvision.transforms as T
        fid = FrechetInceptionDistance(feature=2048).to(device)
        tf = T.Compose([T.Resize((299, 299)), T.PILToTensor()])

        pred_paths = sorted(Path(pred_dir).glob("*.png"))
        gt_paths = sorted([p for p in Path(gt_dir).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

        for p in gt_paths:
            try:
                img = Image.open(p).convert("RGB")
                t = tf(img).unsqueeze(0).to(device)
                fid.update(t, real=True)
            except Exception:
                pass

        for p in pred_paths:
            try:
                img = Image.open(p).convert("RGB")
                t = tf(img).unsqueeze(0).to(device)
                fid.update(t, real=False)
            except Exception:
                pass

        return float(fid.compute().detach().cpu())
    except Exception as e:
        print(f"[Warning] FID disabled: {e}")
        return None
