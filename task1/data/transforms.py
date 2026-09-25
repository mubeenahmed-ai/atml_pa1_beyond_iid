"""Controlled interventions applied on a common 224x224 RGB float tensor in [0,1].

Every function takes / returns tensors of shape (3, H, W) or (B, 3, H, W) so the
*same* transformed pixels can be fed to every backbone (normalisation is applied
later inside each model wrapper).
"""
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def to_common_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    """PIL RGB -> (3, size, size) float in [0,1]; bicubic resize (STL-10 is 96x96)."""
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    return TF.to_tensor(img)


# --------------------------------------------------------------------------- colour
def grayscale(x: torch.Tensor) -> torch.Tensor:
    """Remove chromatic information; keep luminance and therefore all geometry."""
    return TF.rgb_to_grayscale(x, num_output_channels=3)


def hue_rotation(x: torch.Tensor, degrees: float = 120.0) -> torch.Tensor:
    """Rotate hue by a fixed angle on the HSV circle.

    Saturation, value (brightness) and every spatial structure are preserved; only
    *which* colour each pixel has changes, so natural colour statistics are broken
    while colour variety is kept (unlike grayscale).
    """
    hue_factor = (degrees / 360.0) % 1.0
    if hue_factor > 0.5:
        hue_factor -= 1.0
    return TF.adjust_hue(x, hue_factor)


# --------------------------------------------------------------------------- translation
_DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def translate(x: torch.Tensor, delta: int, direction: str) -> torch.Tensor:
    """Shift image content by `delta` pixels using reflection padding + shifted crop.

    direction='right' moves the object to the right (the crop window moves left).
    """
    if delta == 0:
        return x.clone()
    dx, dy = _DIRS[direction]
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    H, W = x.shape[-2:]
    xp = F.pad(x, (delta, delta, delta, delta), mode="reflect")
    top = delta - dy * delta
    left = delta - dx * delta
    out = xp[..., top : top + H, left : left + W]
    return out.squeeze(0) if single else out


# --------------------------------------------------------------------------- patch shuffle
def sample_nonidentity_perm(n: int, gen: torch.Generator) -> torch.Tensor:
    while True:
        p = torch.randperm(n, generator=gen)
        if not torch.equal(p, torch.arange(n)):
            return p


def patch_shuffle(x: torch.Tensor, perm: torch.Tensor, grid: int = 4) -> torch.Tensor:
    """Permute the grid x grid pixel-space patches of a (3,H,W) image with `perm`."""
    C, H, W = x.shape
    ph, pw = H // grid, W // grid
    patches = x.unfold(1, ph, ph).unfold(2, pw, pw)  # C, g, g, ph, pw
    patches = patches.permute(1, 2, 0, 3, 4).reshape(grid * grid, C, ph, pw)
    patches = patches[perm]
    patches = patches.reshape(grid, grid, C, ph, pw).permute(2, 0, 3, 1, 4)
    return patches.reshape(C, grid * ph, grid * pw)


def make_patch_perms(n_images: int, grid: int, seed: int) -> torch.Tensor:
    """One non-identity permutation per image, drawn in fixed order from `seed`."""
    gen = torch.Generator().manual_seed(seed)
    return torch.stack([sample_nonidentity_perm(grid * grid, gen) for _ in range(n_images)])
