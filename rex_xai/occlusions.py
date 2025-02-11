#!/usr/bin/env python
import torch as tt
import numpy as np


def __split_groups(neg_mask):
    # for some reason, it's much faster to do this on the cpu with numpy
    # than it is to use tensor_split
    return np.split(neg_mask, np.where(np.diff(neg_mask) > 1)[0] + 1)


def spectral_occlusion(mask: tt.Tensor, data: tt.Tensor, noise=0.02, device="cpu"):
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
