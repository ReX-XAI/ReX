import copy

import pytest
from rex_xai._utils import Queue, ReXTomlError, Strategy
from rex_xai.config import CausalArgs, process_config_dict, read_config_file
from rex_xai.distributions import Distribution


@pytest.fixture
def non_default_args():
    non_default_args = CausalArgs()
    # rex
    non_default_args.mask_value = "mean"
    non_default_args.seed = 42
    non_default_args.gpu = False
    non_default_args.batch_size = 32
    # rex.onnx
    non_default_args.means = [0.485, 0.456, 0.406]
    non_default_args.stds = [0.229, 0.224, 0.225]
    non_default_args.binary_threshold = 0.5
    non_default_args.norm = 1.0
    # rex.visual
    non_default_args.info = False
    non_default_args.colour = 150
    non_default_args.alpha = 0.1
    non_default_args.raw = True
    non_default_args.resize = True
    non_default_args.progress_bar = False
    non_default_args.grid = True
    non_default_args.mark_segments = True
    non_default_args.heatmap_colours = "viridis"
    non_default_args.multi_style = "separate"
    # causal
    non_default_args.tree_depth = 5
    non_default_args.search_limit = 1000
    non_default_args.iters = 30
    non_default_args.min_box_size = 20
    non_default_args.confidence_filter = 0.5
    non_default_args.weighted = True
    non_default_args.queue_style = Queue.Intersection
    non_default_args.queue_len = 2
    non_default_args.concentrate = True
    # causal.distribution
    non_default_args.distribution = Distribution.BetaBinomial
    non_default_args.blend = 0.5
    non_default_args.distribution_args = [1.1, 1.1]
    # explanation
    non_default_args.chunk_size = 16
    non_default_args.spatial_initial_radius = 20
    non_default_args.spatial_radius_eta = 0.1
    non_default_args.no_expansions = 1
    # explanation.multi
    non_default_args.strategy = Strategy.MultiSpotlight
    non_default_args.spotlights = 5
    non_default_args.spotlight_size = 10
    non_default_args.spotlight_eta = 0.5
    non_default_args.spotlight_step = 10
    non_default_args.max_spotlight_budget = 30
    non_default_args.spotlight_objective_function = "mean"
    non_default_args.permitted_overlap = 0.1
    # explanation.evaluation
    non_default_args.insertion_step = 50
    non_default_args.normalise_curves = False

    return non_default_args


def test_process_config_dict(non_default_args):
    args = CausalArgs()
    config_dict = read_config_file("tests/test_data/rex-test-all-config.toml")

    process_config_dict(config_dict, args)

    assert vars(args) == vars(non_default_args)


def test_process_config_dict_empty():
    args = CausalArgs()
    config_dict = {}
    orig_args = copy.deepcopy(args)

    process_config_dict(config_dict, args)

    assert vars(args) == vars(orig_args)


def test_process_config_dict_invalid_arg(caplog):
    args = CausalArgs()
    config_dict = {'explanation': {'chunk': 10 } }

    process_config_dict(config_dict, args)
    assert caplog.records[0].message == "Invalid or misplaced parameter 'chunk', skipping!"


def test_process_config_dict_invalid_distribution(caplog):
    args = CausalArgs()
    config_dict = {
        "causal": {"distribution": {"distribution": "an-invalid-distribution"}}
    }

    process_config_dict(config_dict, args)

    assert args.distribution == Distribution.Uniform
    assert caplog.records[0].message == "Invalid distribution 'an-invalid-distribution', reverting to default value Distribution.Uniform"


def test_process_config_dict_uniform_distribution():
    args = CausalArgs()
    config_dict = {
        "causal": {
            "distribution": {"distribution": "uniform", "distribution_args": [0.0, 0.0]}
        }
    }

    process_config_dict(config_dict, args)

    assert args.distribution == Distribution.Uniform
    assert args.distribution_args is None


def test_process_config_dict_distribution_args():
    args = CausalArgs()
    config_dict = {
        "causal": {
            "distribution": {
                "distribution": "betabinom",
                "distribution_args": [0.0, 0.0],
            }
        }
    }

    process_config_dict(config_dict, args)

    assert args.distribution == Distribution.BetaBinomial
    assert args.distribution_args == [0.0, 0.0]


def test_process_config_dict_queue_style():
    args = CausalArgs()
    config_dict = {"causal": {"queue_style": "all"}}

    process_config_dict(config_dict, args)
    assert args.queue_style == Queue.All


def test_process_config_dict_queue_style_upper():
    args = CausalArgs()
    config_dict = {"causal": {"queue_style": "ALL"}}

    process_config_dict(config_dict, args)
    assert args.queue_style == Queue.All


def test_process_config_dict_queue_style_invalid(caplog):
    args = CausalArgs()
    config_dict = {"causal": {"queue_style": "an-invalid-queue-style"}}

    process_config_dict(config_dict, args)
    assert args.queue_style == Queue.Area
    assert caplog.records[0].message == "Invalid queue style 'an-invalid-queue-style', reverting to default value Queue.Area"


def test_process_config_dict_strategy():
    args = CausalArgs()
    config_dict = {"explanation": {"multi": {"strategy": "spotlight"}}}

    process_config_dict(config_dict, args)
    assert args.strategy == Strategy.MultiSpotlight


def test_process_config_dict_strategy_invalid(caplog):
    args = CausalArgs()
    config_dict = {"explanation": {"multi": {"strategy": "an-invalid-strategy"}}}

    process_config_dict(config_dict, args)
    assert args.strategy == Strategy.Spatial
    assert caplog.records[0].message == "Invalid strategy 'an-invalid-strategy', reverting to default value Strategy.Spatial"


def test_process_config_dict_blend_invalid():
    args = CausalArgs()
    config_dict = {"causal": {"distribution": {"blend": 20}}}

    with pytest.raises(ReXTomlError):
        process_config_dict(config_dict, args)


def test_process_config_dict_permitted_overlap_invalid():
    args = CausalArgs()
    config_dict = {"explanation": {"multi": {"permitted_overlap": -5}}}

    with pytest.raises(ReXTomlError):
        process_config_dict(config_dict, args)
