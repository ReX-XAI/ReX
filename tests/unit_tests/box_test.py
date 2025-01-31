#!/usr/bin/env python

import numpy as np

from rex_xai.box import initialise_tree, box_dimensions
from rex_xai.distributions import Distribution
from rex_xai.input_data import Data

data_3d = Data(
    input=np.random.rand(1, 64, 64, 64), model_shape=[1, 64, 64, 64], device="cpu"
)
data_2d = Data(input=np.arange(1, 64, 64), model_shape=[1, 64, 64], device="cpu")


def test_data():
    assert data_3d.model_shape == [1, 64, 64, 64]
    assert data_3d.mode == "voxel"
    assert data_2d.model_shape == [1, 64, 64]
    assert data_2d.mode == "spectral"


def test_initialise_tree():
    box_3d = initialise_tree(
        r_lim=64,
        c_lim=64,
        d_lim=64,
        r_start=0,
        c_start=0,
        d_start=0,
        distribution="uniform",
        distribution_args=None,
    )
    box_2d = initialise_tree(
        r_lim=64,
        c_lim=64,
        r_start=0,
        c_start=0,
        distribution="uniform",
        distribution_args=None,
    )

    assert box_3d.depth_start == 0
    assert box_3d.depth_stop == 64

    # Depth does not exist as attribute for 2D boxes
    assert box_2d.depth_start is None
    assert box_2d.depth_stop is None

    assert box_dimensions(box_3d) == (0, 64, 0, 64, 0, 64)
    assert box_dimensions(box_2d) == (0, 64, 0, 64)

    assert (
        box_3d.__repr__()
        == "Box < name: R, row_start: 0, row_stop: 64, col_start: 0, col_stop: 64, depth_start: 0, depth_stop: 64, area: 262144"
    )
    assert (
        box_2d.__repr__()
        == "Box < name: R, row_start: 0, row_stop: 64, col_start: 0, col_stop: 64, area: 4096"
    )

    assert box_3d.shape() == (64, 64, 64)
    assert box_2d.shape() == (64, 64)

    assert box_3d.area() == 262144
    assert box_2d.area() == 4096

    assert box_3d.corners() == (0, 64, 0, 64, 0, 64)
    assert (
        box_2d.corners() == (0, 64, 0, 64)
    )  # TODO: Box_dimensions has the same functionality as corners, should we remove one of them.


def test_spawn_children():
    box_3d = initialise_tree(
        r_lim=64,
        c_lim=64,
        d_lim=64,
        r_start=0,
        c_start=0,
        d_start=0,
        distribution=Distribution.Uniform,
        distribution_args=None,
    )
    box_2d = initialise_tree(
        r_lim=64,
        c_lim=64,
        r_start=0,
        c_start=0,
        distribution=Distribution.Uniform,
        distribution_args=None,
    )

    map_3d = np.zeros((64, 64, 64), dtype="float32")
    maps_2d = np.zeros((64, 64), dtype="float32")

    children_3d = box_3d.spawn_children(min_size=20, mode="voxel", map=map_3d)
    children_2d = box_2d.spawn_children(min_size=2, mode="RGB", map=maps_2d)

    assert len(children_3d) == 4
    assert len(children_2d) == 4

    assert children_3d[0].area() < 262144
    assert children_2d[0].area() < 4096

    total_area_3d = (
        children_3d[0].area()
        + children_3d[1].area()
        + children_3d[2].area()
        + children_3d[3].area()
    )
    assert total_area_3d == 262144

    total_area_2d = (
        children_2d[0].area()
        + children_2d[1].area()
        + children_2d[2].area()
        + children_2d[3].area()
    )
    assert total_area_2d == 4096
