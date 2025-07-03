#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import torch as tt
from scipy.ndimage import gaussian_filter


def __split_groups(neg_mask):
    # for some reason, it's much faster to do this on the cpu with numpy
    # than it is to use tensor_split
    return np.split(neg_mask, np.where(np.diff(neg_mask) > 1)[0] + 1)


def spectral_occlusion(
    mask: tt.Tensor, data: tt.Tensor, noise=0.02, device: str | tt.device = "cpu"
):
    """Linear interpolated occlusion for spectral data, with optional added noise.

    @param mask: boolean valued NDArray
    @param data: data to be occluded
    @param noise: parameter for optional gaussian noise.
        Set to 0.0 if you want simple linear interpolation

    @return torch.Tensor
    """
    neg_mask = tt.where(mask == 0)[0]
    split = __split_groups(neg_mask.detach().cpu().numpy())

    # strangely, this all seems to run faster if we do it on the cpu.
    # TODO Needs further investigation
    local_data = np.copy(data.detach().cpu().numpy())

    for s in split:
        if len(s) <= 1:
            return tt.from_numpy(local_data).to(device)
        start = s[0]
        stop = s[-1]
        dstart = data[:, :, start][0][0].item()
        dstop = data[:, :, stop][0][0].item()
        interp = tt.linspace(dstart, dstop, stop - start)
        if noise > 0.0:
            interp += np.random.normal(0, noise, len(interp))

        local_data[0, 0, start:stop] = interp

    return tt.from_numpy(local_data).to(device)


# Occlusions such as beach, sky, and other context-based occlusions


# Medical-based occlusions could be a CT scan of a healthy patient
def context_occlusion(mask: tt.Tensor, data: tt.Tensor, context: tt.Tensor, noise=0.5):
    """Context based occlusion with optional added noise.

    @param mask: boolean valued NDArray
    @param data: data to be occluded
    @param context: data to be used as occlusion e.g. CT scan of a healthy patient or a road
    @param noise: parameter for optional gaussian noise.
        Set to 0.0 for no noise

    @return torch.Tensor
    """
    if noise > 0.0:
        device = data.device
        context = context.to("cpu")  # As gaussian_filter expects cpu bound
        context = tt.tensor(gaussian_filter(context, sigma=noise), dtype=tt.float32).to(
            device
        )
    return tt.where(mask == False, context, data)
