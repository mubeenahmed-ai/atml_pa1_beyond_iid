"""AdaIN style transfer (Huang & Belongie, 2017).

Architecture follows the public PyTorch re-implementation by Naoto Inoue
(https://github.com/naoto0804/pytorch-AdaIN, MIT licence): a normalised VGG-19
encoder truncated at relu4_1 and a mirrored decoder. Pretrained weights
(`vgg_normalised.pth`, `decoder.pth`) come from that project (downloaded from a
HuggingFace mirror, see README). Only the architecture definition is reused here;
the generation / rejection logic is our own.
"""
import torch
import torch.nn as nn

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),  # relu4-1  <- index 30
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu5-4
)


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    N, C = feat.shape[:2]
    var = feat.reshape(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def adaptive_instance_normalization(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    s_mean, s_std = calc_mean_std(style_feat)
    c_mean, c_std = calc_mean_std(content_feat)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean


def _strip_prefix(sd: dict, prefix: str = "net.") -> dict:
    """Some released checkpoints wrap the Sequential in a `net` attribute."""
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}


class AdaINStyler(nn.Module):
    def __init__(self, vgg_path: str, decoder_path: str, device="cuda"):
        super().__init__()
        vgg.load_state_dict(_strip_prefix(torch.load(vgg_path, map_location="cpu", weights_only=True)))
        decoder.load_state_dict(_strip_prefix(torch.load(decoder_path, map_location="cpu", weights_only=True)))
        self.encoder = nn.Sequential(*list(vgg.children())[:31]).to(device).eval()  # up to relu4_1
        self.decoder = decoder.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """content, style: (B,3,H,W) in [0,1]. Returns stylised image in [0,1]."""
        fc = self.encoder(content)
        fs = self.encoder(style)
        t = adaptive_instance_normalization(fc, fs)
        t = alpha * t + (1 - alpha) * fc
        out = self.decoder(t)
        return out.clamp(0, 1)
