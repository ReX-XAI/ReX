#!/usr/bin/env python

"""generate boxes which together create a occlusion.
This occlusion is realised in the form of a mask over an image"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from anytree import LevelOrderGroupIter, NodeMixin, RenderTree

from rex_xai.mutants.distributions import Distribution, random_coords
from rex_xai.utils.logger import logger


# Enums for different axes
class Axes:
    ROW = 0
    COL = 1
    DEPTH = 2


class BoxInternal:
    """a box is part of an occulsion, a collection of boxes which form a mask over an image"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        row_start,
        row_stop,
        col_start,
        col_stop,
        depth_start=None,  # Optional depth_start and depth_stop for 3D data
        depth_stop=None,
        distribution=None,
        distribution_args=None,
        name="",
    ) -> None:
        self.name = name
        self.distribution = distribution
        self.distribution_args = distribution_args
        self.row_start: int = row_start
        self.row_stop: int = row_stop
        self.col_start: int = col_start
        self.col_stop: int = col_stop
        self.depth_start = depth_start
        self.depth_stop = depth_stop

    def __repr__(self) -> str:
        if self.depth_start is not None and self.depth_stop is not None:
            return (
                f"Box < name: {self.name}, row_start: {self.row_start}, "
                + f"row_stop: {self.row_stop}, col_start: {self.col_start}, "
                + f"col_stop: {self.col_stop}, depth_start: {self.depth_start}, "
                + f"depth_stop: {self.depth_stop}, volume: {self.area()}"
            )
        else:
            return (
                f"Box < name: {self.name}, row_start: {self.row_start}, "
                + f"row_stop: {self.row_stop}, col_start: {self.col_start}, "
                + f"col_stop: {self.col_stop}, area: {self.area()}"
            )

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            return self.name == other
        if isinstance(other, type(self)):
            return NotImplemented
        return self.name == other.name

    def update_name(self, name: str):
        """sets the name of a box"""
        self.name += name

    def shape(self):
        """returns (width, height) of a box if 2d data, else returns (width, height, depth)"""
        if self.depth_start is not None and self.depth_stop is not None:
            return (
                self.row_stop - self.row_start,
                self.col_stop - self.col_start,
                self.depth_stop - self.depth_start,
            )
        else:
            return (self.row_stop - self.row_start, self.col_stop - self.col_start)

    def corners(self):
        """Return (Wstart, Wstop, Hstart, Hstop) of current box if 2d data, else returns (Wstart, Wstop, Hstart, Hstop. Dstart, Dstop)"""
        if self.depth_start is not None and self.depth_stop is not None:
            return (
                self.row_start,
                self.row_stop,
                self.col_start,
                self.col_stop,
                self.depth_start,
                self.depth_stop,
            )
        else:
            return (self.row_start, self.row_stop, self.col_start, self.col_stop)

    def __1d_parts(self):
        if self.col_stop - self.col_start < 4:
            return None

        width = self.col_stop - self.col_start
        xs = random_coords(
            self.distribution, width, 3, self.distribution_args, 1, width
        )
        if xs is None:
            return None

        xs = xs + self.col_start
        ordered = sorted(xs)  # type: ignore

        b0 = Box(
            0,
            0,
            self.col_start,
            ordered[0],
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b0.update_name(":0")

        b1 = Box(
            0,
            0,
            ordered[0],
            ordered[1],
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b1.update_name(":1")

        b2 = Box(
            0,
            0,
            ordered[1],
            ordered[2],
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b2.update_name(":2")

        b3 = Box(
            0,
            0,
            ordered[2],
            self.col_stop,
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b3.update_name(":3")

        return [b0, b1, b2, b3]

    def __2d_parts(self, map=None):
        if self.distribution == Distribution.Adaptive:
            pos = random_coords(self.distribution, self.corners(), map=map)
        else:
            h = int(self.row_stop - self.row_start)
            w = int(self.col_stop - self.col_start)
            space: int = h * w
            pos = random_coords(
                self.distribution, space, 1, self.distribution_args, h, w, map=map
            )

        if pos is None:
            return
        coords = np.unravel_index(
            pos,
            (self.row_stop - self.row_start, self.col_stop - self.col_start),
        )  # type: ignore
        row: int = int(coords[0]) + self.row_start
        col: int = int(coords[1]) + self.col_start

        b0 = Box(
            self.row_start,
            row,
            self.col_start,
            col,
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b0.update_name(":0")

        b1 = Box(
            self.row_start,
            row,
            col,
            self.col_stop,
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b1.update_name(":1")

        b2 = Box(
            row,
            self.row_stop,
            self.col_start,
            col,
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b2.update_name(":2")

        b3 = Box(
            row,
            self.row_stop,
            col,
            self.col_stop,
            distribution=self.distribution,
            distribution_args=self.distribution_args,
            name=self.name,
        )
        b3.update_name(":3")

        return [b0, b1, b2, b3]

    def __3d_parts(self, map=None):
        """
        Create 4 boxes from the original box passed in as an argument.
        Pick two axes to split on randomly using the Axes class and create boxes.
        """

        # Pick two axes to split on randomly using the axes class
        axes = [Axes.ROW, Axes.COL, Axes.DEPTH]
        selected_axes = np.random.choice(axes, 2, replace=False)
        # Range of the different axes to split on
        ranges: Dict = {
            Axes.ROW: [self.row_start, self.row_stop],
            Axes.COL: [self.col_start, self.col_stop],
            Axes.DEPTH: [self.depth_start, self.depth_stop],
        }
        range1 = ranges[selected_axes[0]]
        range2 = ranges[selected_axes[1]]
        logger.debug(
            f"Selected axes: {selected_axes}, which have a range of values, {range1} and {range2}"
        )
        # Get the random coordinates for the two axes
        space = range1[1] - range1[0]
        if space == 0 or space == 1:
            c1 = np.random.choice([range1[0], range1[1]])
        else:
            c1 = random_coords(
                self.distribution,
                space,
                self.distribution_args,
                map=map,
            )
            c1 = range1[0] + c1
        space = range2[1] - range2[0]
        if space == 0 or space == 1:
            c2 = np.random.choice([range2[0], range2[1]])
        else:
            c2 = random_coords(
                self.distribution,
                space,
                self.distribution_args,
                map=map,
            )
            c2 = range2[0] + c2
        logger.debug(f"Random coordinates picked: {c1} and {c2}")
        # If any of the coordinates are None, return None
        if c1 == -1 or c2 == -1:
            return None
        # Create boxes depending on the selected axes
        boxes = []

        subboxes = self.create_box(selected_axes[0], c1, self)
        for i, box in enumerate(subboxes):
            boxes.extend(self.create_box(selected_axes[1], c2, box))
        # Update the names of the boxes
        for i, box in enumerate(boxes):
            box.update_name(f":{i}")
        return boxes

    def create_box(self, axes, c1, box: BoxInternal):
        """
        Create a box depending on the selected axis and the random coordinate
        This will create 2 boxes from the original box passed in as an argument.
        params:
            axes: The axis to split on
            c1: The random coordinate to split on
            box: The original box to split on
        """
        if axes == Axes.ROW:
            b0 = Box(
                box.row_start,
                c1,
                box.col_start,
                box.col_stop,
                box.depth_start,
                box.depth_stop,
                distribution=self.distribution,
                distribution_args=self.distribution_args,
                name=self.name,
            )
            b1 = Box(
                c1,
                box.row_stop,
                box.col_start,
                box.col_stop,
                box.depth_start,
                box.depth_stop,
                distribution=self.distribution,
                distribution_args=self.distribution_args,
                name=self.name,
            )
            return [b0, b1]
        if axes == Axes.COL:
            b0 = Box(
                box.row_start,
                box.row_stop,
                box.col_start,
                c1,
                box.depth_start,
                box.depth_stop,
                distribution=self.distribution,
                distribution_args=self.distribution_args,
                name=self.name,
            )
            b1 = Box(
                box.row_start,
                box.row_stop,
                c1,
                box.col_stop,
                box.depth_start,
                box.depth_stop,
                distribution=self.distribution,
                distribution_args=self.distribution_args,
                name=self.name,
            )
            return [b0, b1]
        if axes == Axes.DEPTH:
            b0 = Box(
                box.row_start,
                box.row_stop,
                box.col_start,
                box.col_stop,
                box.depth_start,
                c1,
                distribution=self.distribution,
                distribution_args=self.distribution_args,
                name=self.name,
            )
            b1 = Box(
                box.row_start,
                box.row_stop,
                box.col_start,
                box.col_stop,
                c1,
                box.depth_stop,
                distribution=self.distribution,
                distribution_args=self.distribution_args,
                name=self.name,
            )
            return [b0, b1]

    def spawn_children(self, min_size, mode, map=None) -> List[Box] | Tuple | None:
        """spawn subboxes from <self>"""
        if self.area() < min_size:
            return ()

        # if we have tabular data, we are of the form (1, x) so we grab 3 random numbers from the
        # distribution and partition with those
        if mode == "tabular" or mode == "spectral":
            return self.__1d_parts()

        # we have a 2 dimension image
        if mode == "RGB" or mode == "L":
            return self.__2d_parts(map=map)

        # we have a 3 dimension data
        if mode == "voxel":
            return self.__3d_parts(map=map)

        return []

    def area(self) -> int:
        """returns the area of a box"""

        if self.depth_start is not None and self.depth_stop is not None:
            return (
                (self.row_stop - self.row_start)
                * (self.col_stop - self.col_start)
                * (self.depth_stop - self.depth_start)
            )

        else:
            if self.row_start == 0 and self.row_stop == 0:
                return self.col_stop - self.col_start
            return (self.row_stop - self.row_start) * (self.col_stop - self.col_start)


class Box(BoxInternal, NodeMixin):
    """An object that represents a box, a set of
    boxes forms an occlusion"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        row_start,
        row_stop,
        col_start,
        col_stop,
        depth_start=None,
        depth_stop=None,
        distribution=None,
        distribution_args=None,
        name="",
        parent=None,
        children=None,
    ) -> None:
        super().__init__(
            row_start,
            row_stop,
            col_start,
            col_stop,
            depth_start,
            depth_stop,
            distribution,
            distribution_args,
            name,
        )
        self.parent = parent
        # anytree nodemixin needs every field to be a tuple, and NoneType object is not iterable,
        # so to get around this limitaton, we set children to be an empty tuple if the box has not spawned any subboxes
        self.children = () if children is None else children

    def add_children_to_tree(self, min_size, mode, map):
        """adds a list of boxes (children) to box tree"""
        if self.children == ():
            try:
                self.children = self.spawn_children(min_size, mode, map)
            except TypeError:
                self.children = ()


def initialise_tree(
    r_lim,
    c_lim,
    distribution,
    distribution_args,
    r_start=0,
    c_start=0,
    d_start=None,
    d_lim=None,
) -> Box:
    """initialise box tree with root node, the whole image"""
    if d_lim is not None and d_start is None:
        d_start = 0
    return Box(
        r_start,
        r_lim,
        c_start,
        c_lim,
        depth_start=d_start,
        depth_stop=d_lim,
        distribution=distribution,
        distribution_args=distribution_args,
        name="R",
    )


def show_tree(tree):
    """Print the box tree to the terminal."""
    print(RenderTree(tree))


def average_box_size(tree, d) -> float:
    """Calculate the average box size at depth <d> of <tree>."""
    areas = [
        [node.area() for node in children] for children in LevelOrderGroupIter(tree)
    ]
    try:
        return np.mean(areas[d], axis=0)
    except IndexError:
        return 0.0


def box_dimensions(
    box: Box,
) -> tuple[int, int, int, int, int, int] | tuple[int, int, int, int]:
    """Returns box dimensions as a 4-tuple or 6-tuple depending on whether the box is 2D or 3D.

    @param box: Box
    @return (int, int, int, int) | (int, int, int, int, int, int)
    """
    if box.depth_start is not None and box.depth_stop is not None:
        return (
            box.row_start,
            box.row_stop,
            box.col_start,
            box.col_stop,
            box.depth_start,
            box.depth_stop,
        )
    else:
        return (box.row_start, box.row_stop, box.col_start, box.col_stop)


def boxes_name_and_dimensions(boxes: List[Box]):
    """Returns a list of all boxes and their dimensions"""
    return [(box.name, box_dimensions(box)) for box in boxes]
