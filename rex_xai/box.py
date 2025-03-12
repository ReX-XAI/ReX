#!/usr/bin/env python

"""generate boxes which together create a occlusion.
This occlusion is realised in the form of a mask over an image"""

from __future__ import annotations

from typing import List, Tuple
from anytree import LevelOrderGroupIter, NodeMixin, RenderTree
import numpy as np

from rex_xai.distributions import Distribution, random_coords


class BoxInternal:
    """a box is part of an occulsion, a collection of boxes which form a mask over an image"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        row_start,
        row_stop,
        col_start,
        col_stop,
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

    def __repr__(self) -> str:
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
        """returns (width, heigh) of a box"""
        return (self.row_stop - self.row_start, self.col_stop - self.col_start)

    def corners(self):
        """Return (Wstart, Wstop, Hstart, Hstop) of current box"""
        return (self.row_start, self.row_stop, self.col_start, self.col_stop)

    def __1d_parts(self):
        c1 = random_coords(self.distribution, [self.col_stop - self.col_start])
        if c1 is not None and isinstance(c1, np.ndarray):
            c1 = c1[0] + self.col_start

        c2 = random_coords(self.distribution, [self.col_stop - self.col_start])
        if c2 is not None and isinstance(c2, np.ndarray):
            c2 = c2[0] + self.col_start

        c3 = random_coords(self.distribution, [self.col_stop - self.col_start])
        if c3 is not None and isinstance(c3, np.ndarray):
            c3 = c3[0] + self.col_start

        if c1 is None or c2 is None or c3 is None:
            return None

        ordered = sorted([c1, c2, c3])

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
                self.distribution, space, h, w, self.distribution_args, map=map
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

        if mode == "voxel":
            pass

        return []

    def area(self) -> int:
        """returns the area of a box"""
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
    r_lim, c_lim, distribution, distribution_args, r_start=0, c_start=0
) -> Box:
    """initialise box tree with root node, the whole image"""
    return Box(
        r_start,
        r_lim,
        c_start,
        c_lim,
        distribution,
        distribution_args,
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


def box_dimensions(box: Box) -> Tuple[int, int, int, int]:
    """Returns box dimensions as a 4-tuple.

    @param box: Box
    @return (int, int, int, int)
    """
    return (box.row_start, box.row_stop, box.col_start, box.col_stop)


def boxes_name_and_dimensions(boxes: List[Box]):
    """Returns a list of all boxes and their dimensions"""
    return [(box.name, box_dimensions(box)) for box in boxes]
