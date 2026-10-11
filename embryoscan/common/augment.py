"""Batched data augmentation on the GPU, shared by the photo and video models.

Training sets live on the GPU as uint8 and are augmented there, a batch at a
time, so there are no DataLoader worker processes (each would load its own
copy of PyTorch, which a 16 GB Windows machine can't afford).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

MEAN = (0.485, 0.456, 0.406)  # ImageNet statistics the pretrained backbones expect
STD = (0.229, 0.224, 0.225)


def to_net(x: torch.Tensor) -> torch.Tensor:
    """(B, 1 or 3, H, W) float in [0, 1] -> (B, 3, H, W) ImageNet-normalised.
    Grayscale input is repeated across the 3 channels."""
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x.expand(-1, 3, -1, -1) - mean) / std


def gpu_augment(x: torch.Tensor, background: float | None) -> torch.Tensor:
    """Random rotation (any angle), flip, zoom, shift, brightness, contrast and
    blur, drawn independently for every image in the batch (all channels of an
    image share the same geometry, so stacked video frames stay aligned).
    Embryos have no natural "up", so every rotation is a valid new example.

    x: (B, C, H, W) uint8 -> (B, C, H, W) float in [0, 1]
    background: fill for rotated-in corners; None uses each image's mean.
    """
    B, _, H, W = x.shape
    dev = x.device
    x = x.float() / 255
    fill = x.mean(dim=(1, 2, 3), keepdim=True) if background is None else background

    def uniform(lo, hi):
        return torch.empty(B, device=dev).uniform_(lo, hi)

    ang = uniform(0, 2 * math.pi)
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    zoom = uniform(0.87, 1.15)
    c, s = ang.cos() / zoom, ang.sin() / zoom
    # affine_grid works in [-1, 1] coordinates; rescale so rotation is in
    # pixel space and a non-square image doesn't get sheared.
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = c * flip
    theta[:, 0, 1] = -s * H / W
    theta[:, 1, 0] = s * flip * W / H
    theta[:, 1, 1] = c
    theta[:, 0, 2] = uniform(-0.08, 0.08)
    theta[:, 1, 2] = uniform(-0.08, 0.08)
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    inside = F.grid_sample(torch.ones_like(x), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    out = out + (1 - inside) * fill

    mean = out.mean(dim=(2, 3), keepdim=True)
    out = (out - mean) * uniform(0.75, 1.25).view(B, 1, 1, 1) + mean + uniform(-0.08, 0.08).view(B, 1, 1, 1)
    blur = torch.rand(B, device=dev) < 0.2
    if blur.any():
        sigma = float(torch.empty(()).uniform_(0.3, 1.2))  # seeded by torch.manual_seed
        out[blur] = TF.gaussian_blur(out[blur], kernel_size=[5, 5], sigma=[sigma, sigma])
    return out.clamp(0, 1)


def dihedral_views(x: torch.Tensor) -> list[torch.Tensor]:
    """The 8 rotations/flips of a batch, for test-time augmentation."""
    out = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        out += [r, torch.flip(r, dims=(3,))]
    return out
