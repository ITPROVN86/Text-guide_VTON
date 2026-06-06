import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.enabled = False
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features[:16]
            self.vgg = vgg.eval()
            for p in self.vgg.parameters():
                p.requires_grad = False
            self.enabled = True
        except Exception:
            self.vgg = None

    def forward(self, pred, target):
        if not self.enabled or self.vgg is None:
            return F.l1_loss(pred, target)
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        self.vgg = self.vgg.to(pred.device)
        return F.l1_loss(self.vgg(pred), self.vgg(target))


def total_variation_loss(flow):
    dx = torch.mean(torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]))
    dy = torch.mean(torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]))
    return dx + dy


class STVTONLoss(nn.Module):
    def __init__(self, lambda_l1=10.0, lambda_perceptual=1.0, lambda_mask=2.0, lambda_tv=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_mask = lambda_mask
        self.lambda_tv = lambda_tv
        self.perc = VGGPerceptualLoss()

    def forward(self, outputs, batch):
        pred = outputs["tryon"]
        target = batch["target"]
        mask_gt = batch["mask"]
        mask_pred = outputs["refined_mask"]
        flow = outputs["flow"]

        l1 = F.l1_loss(pred, target)
        perc = self.perc(pred, target)
        mask = F.binary_cross_entropy(mask_pred.clamp(1e-4, 1-1e-4), mask_gt)
        tv = total_variation_loss(flow)

        total = (
            self.lambda_l1 * l1
            + self.lambda_perceptual * perc
            + self.lambda_mask * mask
            + self.lambda_tv * tv
        )

        return total, {
            "loss_total": float(total.detach().cpu()),
            "loss_l1": float(l1.detach().cpu()),
            "loss_perceptual": float(perc.detach().cpu()),
            "loss_mask": float(mask.detach().cpu()),
            "loss_tv": float(tv.detach().cpu()),
        }
