#!/usr/bin/env python
import numpy as np

from rex_xai.box import box_dimensions


def test_data(data_3d, data_2d):
    assert data_3d.model_shape == [1, 64, 64, 64]
    assert data_3d.mode == "voxel"
    assert data_2d.model_shape == [1, 64, 64]
    assert data_2d.mode == "spectral"

def test_initialise_tree_3d(box_3d):
    assert box_3d.depth_start == 0
    assert box_3d.depth_stop == 64

    assert box_dimensions(box_3d) == (0, 64, 0, 64, 0, 64)

    assert (
            box_3d.__repr__()
            == "Box < name: R, row_start: 0, row_stop: 64, col_start: 0, col_stop: 64, depth_start: 0, depth_stop: 64, volume: 262144"
    )

    assert box_3d.shape() == (64, 64, 64)
    assert box_3d.area() == 262144
    assert box_3d.corners() == (0, 64, 0, 64, 0, 64)


def test_initialise_tree_2d(box_2d):
    # Depth does not exist as attribute for 2D boxes
    assert box_2d.depth_start is None
    assert box_2d.depth_stop is None

    assert box_dimensions(box_2d) == (0, 64, 0, 64)

    assert (
        box_2d.__repr__()
        == "Box < name: R, row_start: 0, row_stop: 64, col_start: 0, col_stop: 64, area: 4096"
    )

    assert box_2d.shape() == (64, 64)
    assert box_2d.area() == 4096

    assert (
        box_2d.corners() == (0, 64, 0, 64)
    )  # TODO: Box_dimensions has the same functionality as corners, should we remove one of them.

def test_spawn_children_3d(box_3d, resp_map_3d):
    # Set seed
    np.random.seed(24)
    children_3d = box_3d.spawn_children(min_size=20, mode="voxel", map=resp_map_3d)
    assert len(children_3d) == 4
    assert children_3d[0].area() < 262144

    total_area_3d = (
            children_3d[0].area()
            + children_3d[1].area()
            + children_3d[2].area()
            + children_3d[3].area()
    )
    assert total_area_3d == 262144

    volumes = [1536, 2560, 96768, 161280]

    # Check splitting of boxes
    row_split, col_split = 1, 24

    row_starts = [0, 0, row_split, row_split]
    row_stops = [row_split, row_split, 64, 64]
    col_starts = [0, col_split, 0, col_split]
    col_stops = [col_split, 64, col_split, 64]

    for i in range(4):
        assert children_3d[i].name == f"R:{i}"
        # Not split through depth
        assert children_3d[i].depth_start == 0
        assert children_3d[i].depth_stop == 64

        assert children_3d[i].area() == volumes[i]

        assert children_3d[i].row_start == row_starts[i]
        assert children_3d[i].row_stop == row_stops[i]
        assert children_3d[i].col_start == col_starts[i]
        assert children_3d[i].col_stop == col_stops[i]



def test_spawn_children_2d(box_2d, resp_map_2d):
    # Set seed
    np.random.seed(24)
    children_2d = box_2d.spawn_children(min_size=2, mode="RGB", map=resp_map_2d)
    assert len(children_2d) == 4
    assert children_2d[0].area() < 4096

    total_area_2d = (
        children_2d[0].area()
        + children_2d[1].area()
        + children_2d[2].area()
        + children_2d[3].area()
    )
    assert total_area_2d == 4096
    print(children_2d)
    areas = [210, 174, 2030, 1682]

    # Check splitting of boxes
    row_split, col_split = 6, 35

    row_starts = [0, 0, row_split, row_split]
    row_stops = [row_split, row_split, 64, 64]
    col_starts = [0, col_split, 0, col_split]
    col_stops = [col_split, 64, col_split, 64]

    for i in range(4):
        assert children_2d[i].name == f"R:{i}"
        assert children_2d[i].area() == areas[i]

        assert children_2d[i].row_start == row_starts[i]
        assert children_2d[i].row_stop == row_stops[i]
        assert children_2d[i].col_start == col_starts[i]
        assert children_2d[i].col_stop == col_stops[i]


