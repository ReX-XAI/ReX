#!/usr/bin/env python3

"""config management"""

import argparse
import importlib.util
import os
from tabulate import tabulate
from os.path import exists, expanduser
from types import ModuleType
from typing import List, Optional, Union

import matplotlib as mpl
import toml  # type: ignore

from rex_xai.mutants.distributions import Distribution, str2distribution
from rex_xai.utils._utils import (
    Queue,
    ReXError,
    ReXPathError,
    ReXTomlError,
    Strategy,
    match_resposnibility_style,
    version,
)
from rex_xai.utils.logger import logger


class Args:
    """args argument object"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods
    def __init__(self) -> None:
        self.config_location: Optional[str] = None
        # input file
        self.path: str = ""
        self.model = None
        self.mode: Optional[str] = None
        self.shape: None = None
        self.db: Optional[str] = None
        # gpu support
        self.gpu: bool = True
        # for reproducability
        self.seed: Union[int, float, None] = None
        # for custom scripts (when used as cmdline tool)
        self.script: Optional[ModuleType] = None
        self.script_location = None
        self.processed = False
        # for custom occlusions through cmdline
        self.context = False
        self.context_location: Optional[str] = (
            None  # Path to the file to use for occlusion
        )
        self.occlusion_noise: Optional[float] = (
            None  # Optional gaussian noise parameter for occlusion
        )
        # onnx processing
        self.means = None
        self.stds = None
        self.norm: Optional[float] = 255.0
        self.binary_threshold = None
        self.intra_op_num_threads = 8
        self.inter_op_num_threads = 8
        self.ort_logger = 3
        # verbosity
        self.verbosity = 1
        # whether to show progress bar or not
        self.progress_bar = True
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
        self.grid = False
        self.heatmap_colours = "magma"
        self.multi_style = "composite"
        # explanation production strategy
        self.no_extract = False
        self.strategy: Strategy = Strategy.Global
        self.complete = False
        self.chunk_size = 10
        self.minimum_confidence_threshold = 0.0
        self.batch_size: int = 1
        self.multi_class: bool = (
            False  # whether to provide multi-class or single class explanations
        )
        # args for spatial strategy
        self.spatial_initial_radius: int = 25
        self.spatial_radius_eta: float = 0.2
        self.no_expansions = 4
        # spotlight args
        self.spotlights: int = 10
        self.spotlight_size: int = 20
        self.spotlight_eta: float = 0.2
        self.spotlight_step: int = 5
        self.spotlight_objective_function: str = "none"
        self.max_spotlight_budget = 40
        self.permitted_overlap: float = 0.0
        # analysis
        self.analyse = None
        self.insertion_step = 100
        self.normalise_curves = True

    def get_dict_repr(self):
        return {
            k.replace("_", " "): v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    def __repr__(self) -> str:
        table = tabulate(
            self.get_dict_repr().items(),
            headers=["Argument", "Value"],
            tablefmt="fancy_grid",
        )
        return f"Args:\n{table}"


class CausalArgs(Args):
    """Creates a causal args object"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        super().__init__()
        self.config_location = None
        self.tree_depth: int = 10
        self.search_limit: Optional[int] = None
        self.mask_value: Union[int, float, str] = 0
        self.confidence_filter = 0.0
        self.min_box_size: int = 10
        self.segmentation = False
        self.data_location: Optional[str] = None
        self.distribution: Distribution = Distribution.Uniform
        self.distribution_args: Optional[List] = None
        self.blend = 0.0
        self.weighted: bool = False
        self.iters = 20
        self.concentrate = False
        self.negative_responsibility = False
        self.use_bounding_box: bool = False
        # queue management
        self.queue_len = 1
        self.queue_style = Queue.Area
        # responsibility
        self.responsibility_style = "multiplicative"

        if self.min_box_size is not None:
            self.chunk_size = self.min_box_size

    def get_dict_repr(self):
        return {
            k.replace("_", " "): v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    def __repr__(self) -> str:
        table = tabulate(
            self.get_dict_repr().items(),
            headers=["Argument", "Value"],
            tablefmt="fancy_grid",
        )
        return f"Args:\n{table}"


def read_config_file(path):
    if path is None:
        return None
    if not os.path.isfile(path):
        raise ReXPathError(path)
    try:
        file_args = toml.load(path)
        return file_args
    except Exception as e:
        raise ReXTomlError(e)


def rex_ascii():
    return (
        "                       @@@@@@@@@@@@@@@                                  \n"
        + "                       @@  @@@@@@@@@@@                                \n"
        + "                     @@@@  @@@@@@@@@@@@@@@@@                          \n"
        + "                     @@@@@@@@@@@@@@@@@@@@@@@                          \n"
        + "                     @@@@@@@@@@@@@@@@@@@@@@@                          \n"
        + "                     @@@@@@@@@@                                       \n"
        + "                     @@@@@@@@@@@@@@@@         %%%                     \n"
        + "   @@              @@@@@@@@@@@                %%%%                    \n"
        + "   @@              @@@@@@@   %%@               %%%%             %%    \n"
        + "   @@            @@@@@@@@ @%%%%%%%              %%%%           %%     \n"
        + "   @@@@        @@@@@@@@@@ %%%%  %%%%%%           %%%%        %%       \n"
        + "   @@@@@@    @@@@@@@@@@@@ %%%%     %%%%%%      %   %%%     %%         \n"
        + "   @@@@@@@@@@@@@@@@@@@@@@ %%%%       %      %%      %%%   %           \n"
        + "     @@@@@@@@@@@@@@@@@@@@ %%%%    %%      %          %%%%             \n"
        + "       @@@@@@@@@@@@@@@@@@ %%%   %%     %%%%%%%%%%%    %%%             \n"
        + "         @@@@@@@@@@@@@@@@ %%%  %%%%   %%%%           % %%%            \n"
        + "         @@@@@@@@@@@@@@@  %%%   %%%%   %%%%        %@   %%%           \n"
        + "           @@@@@@@@@@@@@  %%%    %%%%    %%%     %%      %%%          \n"
        + "             @@@@@@  @@@  %%%     %%%%    %%%% %%         %%%%        \n"
        + "             @@@@        %%%%      %%%%%   %%%%            %%%%  %    \n"
        + "             @@          %%%%        %%%%   %%%%%@          %%%%%     \n"
        + "             @@@@        %            %%      %              %%       \n\n\n"
        + "   Explaining AI through causal Responsibility EXplanations\n"
    )


def cmdargs_parser():
    """parses command line flags"""
    parser = argparse.ArgumentParser(
        prog="ReX",
        description=f"{rex_ascii()}",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "filename",
        help="file to be processed, either an image, a numpy array, a mat file, or a nifti file",
    )
    parser.add_argument(
        "--output",
        nargs="?",
        const="show",
        help="show minimal, sufficient causal explanation, optionally saved to <OUTPUT>. Requires a PIL compatible file extension",
    )

    parser.add_argument(
        "-c", "--config", type=str, help="optional config file to use for ReX"
    )

    parser.add_argument(
        "-n",
        "--no_extract",
        action="store_true",
        help="prevent ReX from extracting an explanation from the responsibility map",
    )

    parser.add_argument(
        "--processed",
        action="store_true",
        help="prevent ReX from performing any preprocessing",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        help="minimum confidence threshold, overriding the setting in <rex.toml>",
    )

    parser.add_argument(
        "--script",
        type=str,
        help="custom loading and preprocessing script, mostly for use with pytorch models",
    )

    parser.add_argument(
        "--context",
        type=str,
        help="custom occlusion path to be loaded and preprocessed, mostly for use with pytorch models",
    )

    parser.add_argument(
        "--noise",
        type=float,
        help="noise level to be added to data used for context occlusion",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="increase verbosity level, either -v or -vv",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="set verbosity level to 0 (errors only), overrides --verbose",
    )

    parser.add_argument(
        "--surface",
        nargs="?",
        const="show",
        help="surface plot of the responsibility map, optionally saved to <SURFACE>",
    )
    parser.add_argument(
        "--heatmap",
        nargs="?",
        const="show",
        help="heatmap plot of the responsibility map, optionally saved to <HEATMAP>",
    )

    parser.add_argument("--model", type=str, help="model in .onnx format")

    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        help="explanation strategy, one of < multi | spatial | global | spotlight >, defaults to global",
    )
    parser.add_argument(
        "--database",
        "-db",
        type=str,
        help="store output in sqlite database <DATABASE>, creating db if necessary. Please include the extension in the filename",
    )

    parser.add_argument(
        "--multi",
        nargs="?",
        const=10,
        help="multiple explanations, with optional number <x> of spotlights, defaults to value in <rex.toml>, or 10 if undefined",
    )

    parser.add_argument(
        "--contrastive",
        action="store_true",
        help="a contrastive explanation: (approximately) minimal, necessary and sufficient",
    )

    parser.add_argument(
        "--complete",
        action="store_true",
        help="a complete explanation: (approximately) minimal, necessary, sufficient and having approximately the same confidence as the original image",
    )

    parser.add_argument(
        "--iters",
        type=int,
        help="manually override the number of iterations set in the config file",
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="area, entropy and (possibly) insertion/deletion curves",
    )

    parser.add_argument(
        "--analyse",
        nargs="?",
        const="print",
        help="area, entropy and (possibly) insertion/deletion curves",
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help="assist ReX with your input type, one of <tabular>, <spectral>, <RGB>, <voxel>, <audio>",
    )

    parser.add_argument(
        "--spectral",
        action="store_true",
        help="set ReX input type to <spectral>, shortcut for --mode spectral",
    )

    parser.add_argument("--version", action="version", version=version())

    return parser


def cmdargs():
    parser = cmdargs_parser()
    args = parser.parse_args()
    return args


def match_strategy(strategy_string):
    """gets explanation extraction strategy"""
    if strategy_string is None:
        return Strategy.Global
    if strategy_string == "multi" or strategy_string == "spotlight":
        return Strategy.MultiSpotlight
    elif strategy_string == "linear" or strategy_string == "global":
        return Strategy.Global
    elif strategy_string == "spatial":
        return Strategy.Spatial
    elif strategy_string == "contrastive":
        return Strategy.Contrastive
    else:
        logger.warning(
            "Invalid strategy '%s', reverting to default value Strategy.Global",
            strategy_string,
        )
    return Strategy.Global


def match_queue_style(qs: str) -> Queue:
    qs = qs.lower()
    if qs == "all":
        return Queue.All
    elif qs == "area":
        return Queue.Area
    elif qs == "dc":
        return Queue.DC
    elif qs == "intersection":
        return Queue.Intersection
    else:
        logger.warning(
            "Invalid queue style '%s', reverting to default value Queue.Area", qs
        )
    return Queue.Area


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
    if cmd_args.quiet:
        args.verbosity = 0
    else:
        args.verbosity = cmd_args.verbose
    if cmd_args.no_extract is True:
        args.no_extract = True
    if cmd_args.database is not None:
        args.db = cmd_args.database
    if cmd_args.mode is not None:
        args.mode = cmd_args.mode
    if cmd_args.spectral:
        args.mode = "spectral"
    if cmd_args.confidence:
        args.minimum_confidence_threshold = cmd_args.confidence

    args.processed = cmd_args.processed


def find_config_path():
    conf_home = expanduser("~/.config/rex.toml")
    # search in current directory first
    if exists("rex.toml"):
        logger.info("using rex.toml in current working directory")
        config_path = "rex.toml"
    # fallback on $HOME/.config on linux/macos
    elif exists(conf_home):
        logger.info(f"using config in {conf_home}")
        config_path = conf_home
    else:
        config_path = None

    return config_path


def apply_dict_to_args(source, args, allowed_values=None):
    for k, v in source.items():
        if type(v) is not dict:
            if allowed_values is not None:
                if k not in allowed_values:
                    logger.warning("Invalid or misplaced parameter '%s', skipping!", k)
                    continue
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                logger.warning(
                    "Parameter '%s' not in CausalArgs attributes, skipping!", k
                )


def validate_numeric_arg_within_bounds(n, lower, upper):
    if n < lower or n > upper:
        raise ReXTomlError(f"Invalid value '{n}': must be between {lower} and {upper}")


def validate_numeric_arg_more_than(n, lower):
    if not n > lower:
        raise ReXTomlError(f"Invalid value '{n}': must be more than {lower}")


def process_config_dict(config_file_args, args):
    expected_values = {
        "rex": ["mask_value", "seed", "gpu", "batch_size"],
        "onnx": [
            "means",
            "stds",
            "binary_threshold",
            "norm",
            "intra_op_num_threads",
            "inter_op_num_threads",
            "ort_logger",
        ],
        "visual": [
            "info",
            "colour",
            "alpha",
            "raw",
            "progress_bar",
            "grid",
            "mark_segments",
            "heatmap_colours",
            "multi_style",
        ],
        "causal": [
            "tree_depth",
            "search_limit",
            "iters",
            "min_box_size",
            "confidence_filter",
            "weighted",
            "negative_responsibility",
            "queue_style",
            "queue_len",
            "concentrate",
            "responsibility_style",
            "use_bounding_box",
        ],
        "distribution": ["distribution", "blend", "distribution_args"],
        "explanation": ["chunk_size", "minimum_confidence_threshold", "multi_class"],
        "spatial": ["spatial_initial_radius", "spatial_radius_eta", "no_expansions"],
        "multi": [
            "strategy",
            "spotlights",
            "spotlight_size",
            "spotlight_eta",
            "spotlight_step",
            "max_spotlight_budget",
            "spotlight_objective_function",
            "permitted_overlap",
        ],
        "evaluation": ["insertion_step", "normalise_curves"],
    }

    if "causal" in config_file_args.keys():
        causal_dict = config_file_args["causal"]
        apply_dict_to_args(causal_dict, args, expected_values["causal"])

        if "distribution" in causal_dict.keys():
            apply_dict_to_args(
                causal_dict["distribution"], args, expected_values["distribution"]
            )

    if "rex" in config_file_args.keys():
        rex_dict = config_file_args["rex"]
        apply_dict_to_args(rex_dict, args, expected_values["rex"])
        for subdict_name in ["visual", "onnx"]:
            if subdict_name in rex_dict.keys():
                apply_dict_to_args(
                    rex_dict[subdict_name], args, expected_values[subdict_name]
                )

    if "explanation" in config_file_args.keys():
        explain_dict = config_file_args["explanation"]
        apply_dict_to_args(explain_dict, args, expected_values["explanation"])
        for subdict_name in ["spatial", "multi", "evaluation"]:
            if subdict_name in explain_dict.keys():
                apply_dict_to_args(
                    explain_dict[subdict_name], args, expected_values[subdict_name]
                )

    if type(args.distribution) is str:
        args.distribution = str2distribution(args.distribution)
        if args.distribution == Distribution.Uniform:
            args.distribution_args = None

    if type(args.queue_style) is str:
        args.queue_style = match_queue_style(args.queue_style)

    if type(args.strategy) is str:
        args.strategy = match_strategy(args.strategy)

    try:
        args.responsibility_style = match_resposnibility_style(
            args.responsibility_style
        )
    except ReXTomlError as e:
        print(e)
        exit()


def process_custom_script(script, args):
    name, _ = os.path.splitext(script)
    spec = importlib.util.spec_from_file_location(name, script)
    script = importlib.util.module_from_spec(spec)  # type: ignore
    try:
        spec.loader.exec_module(script)  # type: ignore
    except Exception as e:
        logger.error("failed to load %s because of %s, exiting...", script, e)
        raise e
    args.script = script
    args.script_location = script


def process_cmd_args(cmd_args, args):
    if cmd_args.script is not None:
        try:
            process_custom_script(cmd_args.script, args)
        except Exception as e:
            logger.fatal(e)
            exit(-1)

    args.path = cmd_args.filename

    args.strategy = match_strategy(cmd_args.strategy)

    if cmd_args.iters is not None:
        args.iters = cmd_args.iters

    if cmd_args.analyse:
        args.analyse = cmd_args.analyse

    if cmd_args.analyze:
        args.analyse = cmd_args.analyse

    if cmd_args.multi is not None:
        args.strategy = Strategy.MultiSpotlight
        args.spotlights = int(cmd_args.multi)
        args.responsibility_style = "additive"

    if cmd_args.contrastive:
        args.strategy = Strategy.Contrastive

    if cmd_args.complete:
        args.strategy = Strategy.Contrastive
        args.complete = True

    if cmd_args.context is not None:
        args.context = True
        if cmd_args.noise is None:
            logger.warning("no noise specified for occlusion so will not be used")
            args.occlusion_noise = 1
        args.occlusion_noise = float(cmd_args.noise)
        args.context_location = cmd_args.context
        args.mask_value = "context"


def load_config(config_path=None):
    if config_path is None:
        config_path = find_config_path()

    default_args = CausalArgs()
    default_args.config_location = config_path

    try:
        config_file_args = read_config_file(config_path)
        if config_file_args is None:
            logger.warning(
                "Could not find a rex.toml, so running with defaults. This might not produce the effect you want..."
            )
            return default_args
        try:
            process_config_dict(config_file_args, default_args)
            return default_args
        except Exception as e:
            logger.warning(
                "exception of type %s: %s, so reverting to default args", type(e), e
            )
            return default_args

    except ReXError as e:
        logger.fatal(e)
        exit(-1)


def get_all_args():
    """parses all arguments from config file and command line"""
    cmd_args = cmdargs()

    config_path = None
    if cmd_args.config is not None:
        config_path = cmd_args.config

    args = load_config(config_path)

    process_cmd_args(cmd_args, args)

    shared_args(cmd_args, args)

    if args.model is None and args.script_location is None:
        raise RuntimeError("either a <model>.onnx or a python file must be provided")

    return args


def validate_args(args: CausalArgs):
    """Validates a CausalArgs object.

    Args:
        args: configuration values for ReX
    """

    # makes sure file exists at path
    if not os.path.isfile(args.path):
        raise FileNotFoundError(f"Input file {args.path} does not exist")

    # make sure if provided with context path then path exists
    if args.context and not os.path.isfile(args.context_location):
        raise FileNotFoundError(f"Context file {args.context_location} does not exist")

    # values that must be between 0 and 1
    for arg in [
        "blend",
        "permitted_overlap",
        "alpha",
        "confidence_filter",
        "spatial_radius_eta",
        "spotlight_eta",
        "binary_threshold",
        "minimum_confidence_threshold",
    ]:
        val = getattr(args, arg)
        if val is not None:
            validate_numeric_arg_within_bounds(val, lower=0.0, upper=1.0)

    # values that must be more than 0
    for arg in [
        "iters",
        "min_box_size",
        "chunk_size",
        "spatial_initial_radius",
        "no_expansions",
        "spotlights",
        "spotlight_size",
        "spotlight_step",
        "max_spotlight_budget",
        "insertion_step",
    ]:
        val = getattr(args, arg)
        if val is not None:
            validate_numeric_arg_more_than(val, lower=0.0)

    # values that must be boolean
    for arg in [
        "gpu",
        "info",
        "progress_bar",
        "raw",
        "grid",
        "mark_segments",
        "weighted",
        "concentrate",
        "normalise_curves",
        "use_bounding_box",
        "multi_class",
    ]:
        val = getattr(args, arg)
        if val is not None:
            if not isinstance(val, bool):
                raise ReXTomlError(f"Invalid value '{val}' for {arg}, must be boolean")

    # custom treatment of specific values
    validate_numeric_arg_within_bounds(args.colour, lower=0.0, upper=255.0)

    if args.multi_style is not None:
        if args.multi_style not in ["composite", "separate"]:
            raise ReXTomlError(
                f"Invalid value '{args.multi_style}' for multi_style, must be 'composite' or 'separate'"
            )

    validate_numeric_arg_more_than(args.queue_len, lower=0.0)
    if not isinstance(args.queue_len, int):
        if args.queue_len != "all":
            raise ReXTomlError(
                f"Invalid value '{args.queue_len}' for queue_len, must be 'all' or an integer"
            )

    if args.distribution_args is not None:
        if not isinstance(args.distribution_args, list):
            raise ReXTomlError(
                f"distribution args must be a list, not '{type(args.distribution_args)}'"
            )
        elif len(args.distribution_args) != 2:
            raise ReXTomlError(
                f"distribution args must be length 2, not {len(args.distribution_args)}"
            )
        if not all([x > 0 for x in args.distribution_args]):
            raise ReXTomlError("All values in distribution args must be more than zero")

    if args.heatmap_colours not in list(mpl.colormaps):
        raise ReXTomlError(
            f"Invalid colourmap '{args.heatmap_colours}', must be a valid matplotlib colourmap"
        )
