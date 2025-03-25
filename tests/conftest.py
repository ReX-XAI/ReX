import numpy as np
import torch as tt
import pytest
from cached_path import cached_path
from rex_xai.utils._utils import get_device
from rex_xai.box import initialise_tree
from rex_xai.config import CausalArgs, process_custom_script, Strategy
from rex_xai.distributions import Distribution
from rex_xai.rex import (
    calculate_responsibility,
    get_prediction_func_from_args,
    load_and_preprocess_data,
    predict_target,
    try_preprocess,
)
from rex_xai.explanation import Explanation
from rex_xai.multi_explanation import MultiExplanation
from syrupy.extensions.amber.serializer import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type

from rex_xai.input_data import Data


@pytest.fixture
def snapshot_explanation(snapshot):
    return snapshot.with_defaults(
        exclude=props(
                "obj_function", # pointer to function that will differ between runs
                "spotlight_objective_function", # pointer to function that will differ between runs
                "script", # path that differs between systems
                "script_location", # path that differs between systems
                "model",
                "target_map", # large array
                "final_mask", # large array
                "explanation" # large array
            ),
            matcher=path_type(
                types=(CausalArgs,),
                replacer=lambda data, _: AmberDataSerializer.object_as_named_tuple( #type: ignore
                    data
                ),  # needed to allow exclude to work for custom classes
            )
    )


@pytest.fixture(scope="session")
def resnet50():
    resnet50_path = cached_path(
        "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/validated/vision/classification/resnet/model/resnet50-v1-7.onnx"
    )
    return resnet50_path


@pytest.fixture(scope="session")
def DNA_model():
    DNA_model_path = cached_path(
        "https://github.com/ReX-XAI/models/raw/6f66a5c0e1480411436be828ee8312e72f0035e1/spectral/simple_DNA_model.onnx"
    )
    return DNA_model_path


@pytest.fixture
def args():
    args = CausalArgs()
    args.path = "tests/test_data/dog.jpg"
    args.iters = 2
    args.search_limit = 1000
    args.gpu = False

    return args


@pytest.fixture
def args_custom(args):
    process_custom_script("tests/scripts/pytorch_resnet50.py", args)
    args.seed = 42

    return args


@pytest.fixture
def args_torch_swin_v2_t(args):
    process_custom_script("tests/scripts/pytorch_swin_v2_t.py", args)
    args.seed = 42

    return args


@pytest.fixture
def args_onnx(args, resnet50):
    args.model = resnet50
    args.seed = 100

    return args


@pytest.fixture
def args_multi(args_custom):
    args = args_custom
    args.path = "tests/test_data/peacock.jpg"
    args.iters = 5
    args.strategy = Strategy.MultiSpotlight
    args.spotlights = 5

    return args


@pytest.fixture
def model_shape(args_custom):
    prediction_func, model_shape = get_prediction_func_from_args(args_custom)

    return model_shape


@pytest.fixture
def prediction_func(args_custom):
    prediction_func, model_shape = get_prediction_func_from_args(args_custom)

    return prediction_func


@pytest.fixture
def model_shape_swin_v2_t(args_torch_swin_v2_t):
    prediction_func, model_shape = get_prediction_func_from_args(args_torch_swin_v2_t)

    return model_shape


@pytest.fixture
def prediction_func_swin_v2_t(args_torch_swin_v2_t):
    prediction_func, model_shape = get_prediction_func_from_args(args_torch_swin_v2_t)

    return prediction_func


@pytest.fixture
def data(args_custom, model_shape, cpu_device):
    data = try_preprocess(args_custom, model_shape, device=cpu_device)
    return data


@pytest.fixture
def data_custom(args_custom, model_shape, cpu_device):
    data = load_and_preprocess_data(model_shape, cpu_device, args_custom)
    data.set_mask_value(args_custom.mask_value)
    return data


@pytest.fixture
def data_multi(args_multi, model_shape, prediction_func, cpu_device):
    data = load_and_preprocess_data(model_shape, cpu_device, args_multi)
    data.set_mask_value(args_multi.mask_value)
    data.target = predict_target(data, prediction_func)
    return data


@pytest.fixture(scope="session")
def cpu_device():
    device = get_device(gpu=False)

    return device


@pytest.fixture
def exp_custom(data_custom, args_custom, prediction_func):
    data_custom.target = predict_target(data_custom, prediction_func)
    maps, run_stats = calculate_responsibility(
        data_custom, args_custom, prediction_func
    )
    exp = Explanation(maps, prediction_func, data_custom, args_custom, run_stats)

    return exp


@pytest.fixture
def exp_onnx(args_onnx, cpu_device):
    prediction_func, model_shape = get_prediction_func_from_args(args_onnx)
    data = load_and_preprocess_data(model_shape, cpu_device, args_onnx)
    data.set_mask_value(args_onnx.mask_value)
    data.target = predict_target(data, prediction_func)
    maps, run_stats = calculate_responsibility(data, args_onnx, prediction_func)
    exp = Explanation(maps, prediction_func, data, args_onnx, run_stats)

    return exp


@pytest.fixture
def exp_extracted(exp_custom):
    exp_custom.extract(Strategy.Global)

    return exp_custom


@pytest.fixture
def exp_multi(args_multi, data_multi, prediction_func):
    maps, run_stats = calculate_responsibility(data_multi, args_multi, prediction_func)
    multi_exp = MultiExplanation(
        maps, prediction_func, data_multi, args_multi, run_stats
    )
    multi_exp.extract(args_multi.strategy)
    return multi_exp


@pytest.fixture
def data_3d():
    voxel = np.zeros((1, 64, 64, 64), dtype=np.float32)
    voxel[0:30, 20:30, 20:35] = 1
    return Data(
        input=voxel,
        model_shape=[1, 64, 64, 64],
        device="cpu",
        mode="voxel"
    )

@pytest.fixture
def data_2d():
    return Data(
        input=np.arange(1, 64, 64),
        model_shape=[1, 64, 64],
        device="cpu"
    )

@pytest.fixture
def box_3d():
    return initialise_tree(
        r_lim=64,
        c_lim=64,
        d_lim=64,
        r_start=0,
        c_start=0,
        d_start=0,
        distribution=Distribution.Uniform,
        distribution_args=None,
    )

@pytest.fixture
def box_2d():
    return initialise_tree(
        r_lim=64,
        c_lim=64,
        r_start=0,
        c_start=0,
        distribution=Distribution.Uniform,
        distribution_args=None,
    )

@pytest.fixture
def resp_map_2d():
    return np.zeros((64, 64), dtype="float32")

@pytest.fixture
def resp_map_3d():
    resp_map = tt.zeros((64, 64, 64), dtype=tt.float32)
    resp_map[0:10, 20:25, 20:35] = 1
    return resp_map