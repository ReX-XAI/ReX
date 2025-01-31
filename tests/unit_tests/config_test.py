import pytest

from rex_xai.config import CausalArgs, process_config_dict, read_config_file
from rex_xai._utils import Strategy, Queue
from rex_xai.distributions import Distribution


@pytest.fixture
def non_default_args():
    non_default_args = CausalArgs()
    # rex
    non_default_args.mask_value = "mean"
    non_default_args.seed = 42
    non_default_args.gpu = False
    non_default_args.batch_size = 32 # NB diff name from toml file
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
    non_default_args.progress_bar = False # NB diff name from toml file
    non_default_args.grid = True
    non_default_args.mark_segments = True
    non_default_args.heatmap_colours = 'viridis' # NB diff name from toml file
    non_default_args.multi_style = 'separate'
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
    non_default_args.distribution_args = [1.1, 1.1] # NB diff name from toml file
    # explanation
    non_default_args.chunk_size = 16 # NB diff name from toml file
    non_default_args.spatial_initial_radius = 20 # NB diff name from toml file
    non_default_args.spatial_radius_eta = 0.1 # NB diff name from toml file
    non_default_args.no_expansions = 1 
    # explanation.multi
    non_default_args.strategy = Strategy.MultiSpotlight # NB diff name from toml file
    non_default_args.spotlights = 5
    non_default_args.spotlight_size = 10
    non_default_args.spotlight_eta = 0.5
    non_default_args.spotlight_step = 10
    non_default_args.max_spotlight_budget = 30
    non_default_args.spotlight_objective_function = 'mean'
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
