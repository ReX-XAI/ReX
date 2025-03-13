import pytest
from rex_xai._utils import Strategy
from rex_xai.config import CausalArgs, shared_args, cmdargs_parser, process_cmd_args


# filename (path)
# output (path)
# config (path)
# processed (bool)
# script (path)
# verbose (int)
# quiet (bool)
# surface (path)
# heatmap (path)
# model (path)
# strategy (multi | spatial | global | spotlight)
# database (path)
# multi (int)
# contrastive (int)
# iters (int)
# analyze/analyse (bool)
# mode (<tabular>, <spectral>, <RGB>, <voxel>, <audio>)
# spectral (bool)

@pytest.fixture
def non_default_cmd_args():
    return ['filename.jpg',
        '--output', 'output_path.jpg',
        '--config', 'path/to/rex.toml',
        '--processed',
        '--script', 'tests/scripts/pytorch_resnet50.py',
        '-vv',
        '--surface', 'surface_path.jpg',
        '--heatmap', 'heatmap_path.jpg',
        '--model', 'path/to/model.onnx',
        '--strategy', 'multi',
        '--database', 'path/to/database.db',
        '--multi', '5',
        '--iters', '10',
        '--analyze',
        '--mode', 'RGB',
    ]


def test_process_cmd_args(non_default_cmd_args):
    args = CausalArgs()
    parser = cmdargs_parser()
    cmd_args = parser.parse_args(non_default_cmd_args)
    process_cmd_args(cmd_args, args)

    #assert args.script == cmd_args.script
    assert args.path == cmd_args.filename
    assert args.strategy == Strategy.MultiSpotlight
    assert args.iters == int(cmd_args.iters)
    assert args.analyze 
    assert args.spotlights == int(cmd_args.multi)

    
def test_process_shared_args(non_default_cmd_args):
    args = CausalArgs()
    parser = cmdargs_parser()
    cmd_args = parser.parse_args(non_default_cmd_args)
    shared_args(cmd_args, args)

    
    assert args.config_location == cmd_args.config
    assert args.model == cmd_args.model
    assert args.surface == cmd_args.surface
    assert args.heatmap == cmd_args.heatmap
    assert args.output == cmd_args.output
    assert args.verbosity == cmd_args.verbose
    assert args.db == cmd_args.database
    assert args.mode == cmd_args.mode
    assert args.processed == cmd_args.processed

