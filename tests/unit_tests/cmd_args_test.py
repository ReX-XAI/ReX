from types import ModuleType

import pytest
from rex_xai._utils import Strategy
from rex_xai.config import CausalArgs, cmdargs_parser, process_cmd_args, shared_args

# contrastive (int)
# spectral (bool)


@pytest.fixture
def non_default_cmd_args():
    args_list = [
        "filename.jpg",
        "--output",
        "output_path.jpg",
        "--config",
        "path/to/rex.toml",
        "--processed",
        "--script",
        "tests/scripts/pytorch_resnet50.py",
        "-vv",
        "--surface",
        "surface_path.jpg",
        "--heatmap",
        "heatmap_path.jpg",
        "--model",
        "path/to/model.onnx",
        "--strategy",
        "multi",
        "--database",
        "path/to/database.db",
        "--multi",
        "5",
        "--iters",
        "10",
        "--analyze",
        "--mode",
        "RGB",
    ]
    parser = cmdargs_parser()
    cmd_args = parser.parse_args(args_list)

    return cmd_args


def test_process_cmd_args(non_default_cmd_args):
    args = CausalArgs()
    process_cmd_args(non_default_cmd_args, args)

    assert isinstance(args.script, ModuleType)
    assert args.path == non_default_cmd_args.filename
    assert args.strategy == Strategy.MultiSpotlight
    assert args.iters == int(non_default_cmd_args.iters)
    assert args.analyze
    assert args.spotlights == int(non_default_cmd_args.multi)


def test_process_shared_args(non_default_cmd_args):
    args = CausalArgs()
    shared_args(non_default_cmd_args, args)

    assert args.config_location == non_default_cmd_args.config
    assert args.model == non_default_cmd_args.model
    assert args.surface == non_default_cmd_args.surface
    assert args.heatmap == non_default_cmd_args.heatmap
    assert args.output == non_default_cmd_args.output
    assert args.verbosity == non_default_cmd_args.verbose
    assert args.db == non_default_cmd_args.database
    assert args.mode == non_default_cmd_args.mode
    assert args.processed == non_default_cmd_args.processed


def test_quiet_overrides_verbose():
    cmd_args_list = ["filename.jpg", "-vv", "--quiet"]
    parser = cmdargs_parser()
    cmd_args = parser.parse_args(cmd_args_list)
    args = CausalArgs()
    shared_args(cmd_args, args)

    assert args.verbosity == 0


def test_contrastive():
    cmd_args_list = ["filename.jpg", "--contrastive", "5"]
    parser = cmdargs_parser()
    cmd_args = parser.parse_args(cmd_args_list)
    args = CausalArgs()
    process_cmd_args(cmd_args, args)
    
    assert args.strategy == Strategy.Contrastive
    assert args.spotlights == int(cmd_args.contrastive)