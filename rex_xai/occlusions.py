#!/usr/bin/env python
import torch as tt
import numpy as np
from rex_xai.input_data import Data


def set_mask_value(m, data: Data, device="cpu"):
    assert data.data is not None
    # if m is a number, then if might still need to be normalised
    if isinstance(m, (int, float)):
        return m
    if m == "min":
        return tt.min(data.data).item()  # type: ignore
    if m == "mean":
        return tt.mean(data.data).item()  # type: ignore
    if m == "spectral":
        return lambda m, d: spectral_occlusion(m, d, device=device)


def __split_groups(neg_mask):
    # for some reason, it's much faster to do this on the cpu with numpy
    # than it is to use tensor_split
    return np.split(neg_mask, np.where(np.diff(neg_mask) > 1)[0] + 1)


def spectral_occlusion(mask: tt.Tensor, data: tt.Tensor, noise=0.03, device="cpu"):
    """Linear interpolated occlusion for spectral data, with optional added noise.

    @param mask: boolean valued NDArray
    @param data: data to be occluded
    @param noise: parameter for optional gaussian noise.
        Set to 0.0 if you want simple linear interpolation

    @return torch.Tensor
    """
    # we want all groups of False in the mask, as tt.where(neg_mask == 0)[0] is
    # all 0s (as we only have one row), we ignore it
    neg_mask = tt.where(mask == 0)[1]
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
