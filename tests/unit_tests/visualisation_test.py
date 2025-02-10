import os

import torch as tt
from rex_xai.visualisation import save_image, voxel_plot

from rex_xai.config import CausalArgs


def test_surface(exp_extracted, tmp_path):
    p = tmp_path / "surface.png"
    exp_extracted.surface_plot(path=p)

    assert os.path.exists(p)
    assert os.stat(p).st_size > 0


def test_heatmap(exp_extracted, tmp_path):
    p = tmp_path / "heatmap.png"
    exp_extracted.heatmap_plot(path=p)

    assert os.path.exists(p)
    assert os.stat(p).st_size > 0


def test_save_exp(exp_extracted, tmp_path):
    p = tmp_path / "exp.png"
    exp_extracted.save(path=p)

    assert os.path.exists(p)
    assert os.stat(p).st_size > 0

def test_save_image_3d(data_3d):
    # Explanation mask for the voxel data - random values of 0s and 1s
    explanation = tt.zeros((1, 64, 64, 64), dtype=tt.bool, device="cpu")
    explanation[0, 32:64, 32:64, 32:64] = 1
    args = CausalArgs()
    args.mode = "voxel"
    args.output = "test.png"
    save_image(explanation, data_3d, args, path=args.output)
    assert os.path.exists(args.output)
    assert os.path.getsize(args.output) > 0

    os.remove(args.output)


def test_voxel_plot(data_3d):
    resp_map = tt.zeros((64, 64, 64), dtype=tt.float32)
    resp_map[0:10, 20:25, 20:35] = 1
    args = CausalArgs()
    # Create a cube in data
    voxel_plot(args, resp_map, data_3d, path="test.png")
    assert os.path.exists("test.png")
    assert os.path.getsize("test.png") > 0
    os.remove("test.png")
