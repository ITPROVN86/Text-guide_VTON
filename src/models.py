import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers += [
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TextCondition(nn.Module):
    def __init__(self, text_dim=512, out_ch=64):
        super().__init__()
        self.text_dim = text_dim
        self.proj = nn.Sequential(
            nn.Linear(text_dim, out_ch),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch, out_ch),
        )

    def encode_text_hash(self, captions, device):
        vecs = []
        for cap in captions:
            g = torch.Generator(device="cpu")
            seed = abs(hash(cap)) % (2**31)
            g.manual_seed(seed)
            v = torch.randn(self.text_dim, generator=g)
            vecs.append(v)
        return torch.stack(vecs, dim=0).to(device)

    def forward(self, captions, h, w, device):
        v = self.encode_text_hash(captions, device)
        v = self.proj(v).unsqueeze(-1).unsqueeze(-1)
        return v.expand(-1, -1, h, w)


class FlowWarpNet(nn.Module):
    def __init__(self, in_ch=3+3+1+64, base=64):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = Down(base, base*2)
        self.e3 = Down(base*2, base*4)
        self.e4 = Down(base*4, base*8)
        self.u3 = Up(base*8, base*4, base*4)
        self.u2 = Up(base*4, base*2, base*2)
        self.u1 = Up(base*2, base, base)
        self.flow_head = nn.Sequential(nn.Conv2d(base, 2, 3, padding=1), nn.Tanh())

    def forward(self, person, cloth, mask, text_map):
        x = torch.cat([person, cloth, mask, text_map], dim=1)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d3 = self.u3(e4, e3)
        d2 = self.u2(d3, e2)
        d1 = self.u1(d2, e1)
        flow = self.flow_head(d1) * 0.25
        warped = self.warp(cloth, flow)
        return warped, flow

    @staticmethod
    def warp(x, flow):
        b, c, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, h, w, 2)
        flow_grid = flow.permute(0, 2, 3, 1)
        warped_grid = grid + flow_grid
        return F.grid_sample(x, warped_grid, mode="bilinear", padding_mode="border", align_corners=True)


class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.InstanceNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.InstanceNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.relu(self.w_g(g) + self.w_x(x)))
        return x * alpha


class AttentionUNetMaskRefiner(nn.Module):
    def __init__(self, in_ch=3+3+1+64, base=64):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = Down(base, base*2)
        self.e3 = Down(base*2, base*4)
        self.e4 = Down(base*4, base*8)

        self.ag3 = AttentionGate(base*8, base*4, base*2)
        self.ag2 = AttentionGate(base*4, base*2, base)
        self.ag1 = AttentionGate(base*2, base, max(1, base//2))

        self.u3 = Up(base*8, base*4, base*4)
        self.u2 = Up(base*4, base*2, base*2)
        self.u1 = Up(base*2, base, base)
        self.out = nn.Sequential(nn.Conv2d(base, 1, 1), nn.Sigmoid())

    def forward(self, person, warped_cloth, mask, text_map):
        x = torch.cat([person, warped_cloth, mask, text_map], dim=1)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        e3a = self.ag3(e4, e3)
        d3 = self.u3(e4, e3a)

        e2a = self.ag2(d3, e2)
        d2 = self.u2(d3, e2a)

        e1a = self.ag1(d2, e1)
        d1 = self.u1(d2, e1a)
        return self.out(d1)


class SynthNet(nn.Module):
    def __init__(self, in_ch=3+3+1+64, base=64):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = Down(base, base*2)
        self.e3 = Down(base*2, base*4)
        self.e4 = Down(base*4, base*8)
        self.u3 = Up(base*8, base*4, base*4)
        self.u2 = Up(base*4, base*2, base*2)
        self.u1 = Up(base*2, base, base)
        self.out = nn.Sequential(nn.Conv2d(base, 3, 3, padding=1), nn.Tanh())

    def forward(self, person, warped_cloth, refined_mask, text_map):
        x = torch.cat([person, warped_cloth, refined_mask, text_map], dim=1)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d3 = self.u3(e4, e3)
        d2 = self.u2(d3, e2)
        d1 = self.u1(d2, e1)
        gen = self.out(d1)
        out = refined_mask * gen + (1 - refined_mask) * person
        return out


class STVTON(nn.Module):
    def __init__(self, base_channels=64, text_dim=512, use_text_condition=True):
        super().__init__()
        self.use_text_condition = use_text_condition
        self.text = TextCondition(text_dim=text_dim, out_ch=64)
        in_ch = 3 + 3 + 1 + 64
        self.warp_net = FlowWarpNet(in_ch=in_ch, base=base_channels)
        self.mask_refiner = AttentionUNetMaskRefiner(in_ch=in_ch, base=base_channels)
        self.synth = SynthNet(in_ch=in_ch, base=base_channels)

    def forward(self, person, cloth, mask, captions):
        b, _, h, w = person.shape
        if self.use_text_condition:
            text_map = self.text(captions, h, w, person.device)
        else:
            text_map = torch.zeros(b, 64, h, w, device=person.device)

        warped_cloth, flow = self.warp_net(person, cloth, mask, text_map)
        refined_mask = self.mask_refiner(person, warped_cloth, mask, text_map)
        tryon = self.synth(person, warped_cloth, refined_mask, text_map)
        return {
            "tryon": tryon,
            "warped_cloth": warped_cloth,
            "refined_mask": refined_mask,
            "flow": flow,
        }
