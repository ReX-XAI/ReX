#!/usr/bin/env python3

"""config management"""

import sys
from typing import List, Optional
from types import ModuleType
import argparse
from enum import Enum
import os
from os.path import exists, expanduser
import importlib.util
import numpy as np
import toml  # type: ignore


from rex_ai.distributions import str2distribution
from rex_ai.prediction import Prediction
from rex_ai.distributions import Distribution

CAUSAL = Enum("CAUSAL", ["Responsibility"])

Strategy = Enum(
    "Strategy", ["Global", "Spatial", "Spotlight", "MultiSpotlight"]
)

Queue = Enum("Queue", ["Area", "All", "Intersection", "DC"])


class Args:
    """args argument object"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods
    def __init__(self) -> None:
        self.config_location: Optional[str] = None
        # input file
        self.path: Optional[str] = None
        self.model = None
        self.mode = None
        self.shape: None = None
        self.db: Optional[str] = None
        # gpu support
        self.gpu: bool = True
        # for reproducability
        self.seed: Optional[None] = None
        # for custom scripts (when used as cmdline tool)
        self.custom: Optional[ModuleType] = None
        self.custom_location = None
        self.processed = False
        # min-max normalization
        self.means = None
        self.stds = None
        self.binary_threshold = None
        # verbosity
        self.verbosity = 0
        # whether to show progress bar or not
        self.progress = True
        # save explanation to output
        self.output = None
        self.surface: Optional[str] = None
        self.heatmap: Optional[str] = None
        self.info = True
        self.raw: bool = False
        self.colour: int = 200
        self.mark_segments = False
        self.alpha = 0.2
        self.all = False
        self.resize = False
        self.grid = False
        self.heatmap_colours = 'magma'
        # explanation production strategy
        self.strategy: Strategy = Strategy.Spatial
        self.chunk_size = 25
        self.batch: int = 1
        # args for spatial strategy
        self.spatial_radius: int = 25
        self.spatial_eta: float = 0.2
        self.no_expansions = 50
        # spotlight args
        self.spotlights: int = 10
        self.spotlight_size: int = 20
        self.spotlight_eta: float = 0.2
        self.spotlight_step: int = 5
        self.spotlight_objective_function = np.mean
        # analysis
        self.analyze: bool = False
        self.insertion_step = 100

    def __repr__(self) -> str:
        return (
            f"Args <file: {self.path}, model: {self.model}, "
            + f"gpu: {self.gpu}, "
            + f"mode: {self.mode}, "
            + f"progress_bar: {self.progress}, "
            + f"output_file: {self.output}, surface_plot: {self.surface}, "
            + f"heatmap_plot: {self.heatmap}, "
            + f"means: {self.means}, stds: {self.stds}, "
            + f"explanation_strategy: {self.strategy}, "
            + f"chunk size: {self.chunk_size}, "
            + f"spatial_radius: {self.spatial_radius}, "
            + f"spatial_eta: {self.spatial_eta}, seed: {self.seed}, "
            + f"db: {self.db}, "
            + f"custom: {self.custom}, verbosity: {self.verbosity}, "
            + f"spotlights: {self.spotlights}, spotlight_size: {self.spotlight_size}, "
            + f"spotlight_eta: {self.spotlight_eta}, "
            + f"no_expansions: {self.no_expansions}, "
            + f"obj_function: {self.spotlight_objective_function}, "
        )


class CausalArgs(Args):
    """Creates a causal args object"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        super().__init__()
        self.config_location = None
        self.type = CAUSAL
        self.tree_depth: int = 10
        self.search_limit: Optional[int] = None
        self.mask_value = 0
        self.confidence_filter = 0.0
        self.min_box_size: int = 10
        self.segmentation = False
        self.data_location: Optional[str] = None
        self.target: Optional[Prediction] = None
        self.distribution: Distribution = Distribution.Uniform
        self.distribution_args: Optional[List] = None
        self.blend = 0.0
        self.weighted: bool = False
        self.iters = 30
        self.concentrate = False
        # queue management
        self.queue_len = 1
        self.queue_style = Queue.Area

    def __repr__(self) -> str:
        return (
            "Causal Args <"
            + Args.__repr__(self)
            + f"mask_value: {self.mask_value}, "
            + f"tree_depth: {self.tree_depth}, search_limit: {self.search_limit}, "
            + f"min_box_size: {self.min_box_size}, weighted: {self.weighted}, "
            + f"confidence_filter: {self.confidence_filter}, "
            + f"data_locations: {self.data_location}, distribution: {self.distribution}, "
            + f"distribution_args: {self.distribution_args}, "
            + f"queue_len: {self.queue_len}, queue_style {self.queue_style}, "
            + f"concentrate: {self.concentrate}, "
            + f"iterations: {self.iters}>"
        )


def get_config_file(path):
    """parses toml file into dictionary"""
    try:
        file_args = toml.load(path)
        return file_args
    except Exception:
        return FileNotFoundError


def cmdargs():
    """parses command line flags"""
    parser = argparse.ArgumentParser(
        prog="ReX",
        description="Explaining AI through causal reasoning",
    )
    parser.add_argument(
        "filename",
        help="file to be processed, assumes that file is 3 channel (RGB or BRG)",
    )
    parser.add_argument(
        "--output",
        nargs="?",
        const="show",
        help="show minimal explanation, optionally saved to <OUTPUT>. Requires a PIL compatible file extension",
    )
    parser.add_argument(
        "-c", "--config", type=str, help="config file to use for rex"
    )

    parser.add_argument(
        "--processed",
        action="store_true",
        help="don't perform any processing with rex itself",
    )

    parser.add_argument(
        "--script", type=str, help="custom loading and preprocessing script"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="verbosity level, either -v or -vv, or -vvv",
    )
    parser.add_argument(
        "--surface",
        nargs="?",
        const="show",
        help="surface plot, optionally saved to <SURFACE>",
    )
    parser.add_argument(
        "--heatmap",
        nargs="?",
        const="show",
        help="heatmap plot, optionally saved to <HEATMAP>",
    )

    parser.add_argument("--model", type=str, help="model, must be onnx format")

    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        help="explanation strategy, one of < multi | spatial | linear | spotlight >",
    )
    parser.add_argument(
        "--database",
        "-db",
        type=str,
        help="store output in sqlite database <DATABASE>, creating db if necessary",
    )
    parser.add_argument(
        "--iters",
        type=int,
        help="manually override the number of iterations set in the config file",
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="area, entropy different and insertion/deletion curves",
    )
    parser.add_argument(
        "--analyse",
        action="store_true",
        help="area, entropy different and insertion/deletion curves",
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
        help="produce a complete breakdown of the image",
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help="assist ReX with your input type, one of <tabular>, <spectral>, <RGB>, <L>, <voxel>, <audio>",
    )

    args = parser.parse_args()
    return args


def match_strategy(cmd_args):
    """gets explanation extraction strategy"""
    if cmd_args.strategy == "multi":
        return Strategy.MultiSpotlight
    if cmd_args.strategy == "linear" or cmd_args.strategy == "global":
        return Strategy.Global
    if cmd_args.strategy == "spotlight":
        return Strategy.Spotlight
    if cmd_args.strategy == "spatial":
        pass
    return Strategy.Spatial


def match_queue_style(qs: str) -> Queue:
    qs = qs.lower()
    if qs == "all":
        return Queue.All
    if qs == "area":
        return Queue.Area
    if qs == "dc":
        return Queue.DC
    return Queue.Intersection


def get_objective_function(multi_dict):
    """gets objective function for spotlight search"""
    try:
        f = multi_dict["obj_function"]
        if f == "mean":
            return np.mean
        if f == "max":
            return np.max
        if f == "min":
            return np.min
    except KeyError:
        pass
    return np.mean


def shared_args(cmd_args, args: CausalArgs):
    """parses shared args"""
    if cmd_args.config is not None:
        args.config_location = cmd_args.config
    if cmd_args.model is not None:
        args.model = cmd_args.model
    if cmd_args.surface is not None:
        args.surface = cmd_args.surface
    if cmd_args.heatmap is not None:
        args.heatmap = cmd_args.heatmap
    if cmd_args.output is not None:
        args.output = cmd_args.output
    if cmd_args.verbose > 0:
        args.verbosity = cmd_args.verbose
    if cmd_args.database is not None:
        args.db = cmd_args.database
    if cmd_args.mode is not None:
        args.mode = cmd_args.mode

    args.processed = cmd_args.processed


def get_all_args(path=None):
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    """parses all arguments from config file and command line"""
    cmd_args = cmdargs()

    path = None
    if cmd_args.config is not None:
        path = cmd_args.config
    else:
        conf_home = expanduser("~/.config/rex.toml")
        # search in current directory first
        if exists("rex.toml"):
            print("using rex.toml in current working directory")
            path = "rex.toml"
        # fallback on $HOME/.config on linux/macos
        elif exists(conf_home):
            print(f"using config in {conf_home}")
            path = conf_home

    args = CausalArgs()
    args.config_location = path

    try:
        config_file_args = get_config_file(path)

        causal_dict = config_file_args["causal"]
        # print(causal_dict)
        if "tree_depth" in causal_dict:
            args.tree_depth = causal_dict["tree_depth"]
        if "search_limit" in causal_dict:
            args.search_limit = causal_dict["search_limit"]
        if "weighted" in causal_dict:
            args.weighted = causal_dict["weighted"]
        if "queue_len" in causal_dict:
            ql = causal_dict["queue_len"]
            if ql != "all":
                args.queue_len = causal_dict["queue_len"]
        if "queue_style" in causal_dict:
            args.queue_style = match_queue_style(causal_dict["queue_style"])
        if "iters" in causal_dict:
            args.iters = causal_dict["iters"]
        if "min_box_size" in causal_dict:
            args.min_box_size = causal_dict["min_box_size"]
        if "confidence_filter" in causal_dict:
            args.confidence_filter = causal_dict["confidence_filter"]
        if "segmentation" in causal_dict:
            args.segmentation = causal_dict["segmentation"]
        if "concentrate" in causal_dict:
            args.concentrate = causal_dict["concentrate"]

        dist = causal_dict["distribution"]
        d = dist["distribution"]
        args.distribution = str2distribution(d)
        if "dist_args" in dist:
            args.distribution_args = dist["dist_args"]
        if "blend" in dist:
            b = dist["blend"]
            if b < 0.0 or b > 1.0:
                print("impossible blend value")
                sys.exit(-1)
            args.blend = dist["blend"]

        rex_dict = config_file_args["rex"]
        if "mask_value" in rex_dict:
            args.mask_value = rex_dict["mask_value"]
        if "seed" in rex_dict:
            args.seed = rex_dict["seed"]
        if "gpu" in rex_dict:
            args.gpu = rex_dict["gpu"]
        if "batch_size" in rex_dict:
            args.batch = rex_dict["batch_size"]

        if "onnx" in rex_dict:
            onnx = rex_dict["onnx"]
            if "means" in onnx:
                args.means = onnx["means"]
            if "stds" in onnx:
                args.stds = onnx["stds"]
            if "binary_threshold" in onnx:
                args.binary_threshold = onnx["binary_threshold"]


        if "visual" in rex_dict:
            if "info" in rex_dict["visual"]:
                args.info = rex_dict["visual"]["info"]
            if "colour" in rex_dict["visual"]:
                args.colour = rex_dict["visual"]["colour"]
            if "color" in rex_dict["visual"]:
                args.colour = rex_dict["visual"]["color"]
            if "alpha" in rex_dict["visual"]:
                args.alpha = rex_dict["visual"]["alpha"]
            if "raw" in rex_dict["visual"]:
                args.raw = rex_dict["visual"]["raw"]
            if "resize" in rex_dict["visual"]:
                args.resize = rex_dict["visual"]["resize"]
            if "progress_bar" in rex_dict["visual"]:
                args.progress = rex_dict["visual"]["progress_bar"]
            if "grid" in rex_dict["visual"]:
                args.grid = rex_dict["visual"]["grid"]
            if "mark_segments" in rex_dict["visual"]:
                args.mark_segments = rex_dict["visual"]["mark_segments"]
            if "heatmap" in rex_dict["visual"]:
                args.heatmap_colours = rex_dict["visual"]["heatmap"]

        explain_dict = config_file_args["explanation"]
        if "chunk" in explain_dict:
            args.chunk_size = explain_dict["chunk"]

        # spatial args
        spatial_dict = explain_dict["spatial"]
        if "initial_radius" in spatial_dict:
            args.spatial_radius = spatial_dict["initial_radius"]
        if "radius_eta" in spatial_dict:
            args.spatial_eta = spatial_dict["radius_eta"]
        if "no_expansions" in spatial_dict:
            args.no_expansions = spatial_dict["no_expansions"]

        multi_dict = explain_dict["multi"]
        if "spotlights" in multi_dict:
            args.spotlights = multi_dict["spotlights"]
        if "spotlight_size" in multi_dict:
            args.spotlight_size = multi_dict["spotlight_size"]
        if "spotlight_eta" in multi_dict:
            args.spotlight_eta = multi_dict["spotlight_eta"]
        if "spotlight_step" in multi_dict:
            args.spotlight_step = multi_dict["spotlight_step"]
        args.spotlight_objective_function = get_objective_function(multi_dict)  # type: ignore


        eval_dict = explain_dict["evaluation"]
        if "insertion_step" in eval_dict:
            args.insertion_step = eval_dict["insertion_step"]

    except KeyError as e:
        print(f"key error {e} in {path}, so reverting to default args")

    except TypeError:
        print(
            "could not find a rex.toml, so running with defaults. This might not produce the effect you want..."
        )

    if cmd_args.script is not None:
        try:
            name, _ = os.path.splitext(cmd_args.script)
            spec = importlib.util.spec_from_file_location(name, cmd_args.script)
            script = importlib.util.module_from_spec(spec)  # type: ignore
            try:
                spec.loader.exec_module(script)  # type: ignore
            except Exception as e:
                print(f"failed to load {name} because of {e}")
            args.custom = script
            args.custom_location = cmd_args.script
        except ImportError:
            pass

    if args.distribution == Distribution.Uniform:
        args.distribution_args = None

    args.path = cmd_args.filename

    shared_args(cmd_args, args)

    args.strategy = match_strategy(cmd_args)

    if args.model is None and args.custom_location is None:
        print("either a <model>.onnx or a python file must be provided")
        sys.exit(-1)

    if cmd_args.iters is not None:
        args.iters = cmd_args.iters


    if cmd_args.analyze or cmd_args.analyse:
        args.analyze = True

    if cmd_args.show_all:
        args.all = True

    return args
