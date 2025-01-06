import pytest

from cached_path import cached_path

from rex_xai.config import CausalArgs, process_custom_script
from rex_xai._utils import get_device
from rex_xai.explanation import (
    try_preprocess,
    get_prediction_func_from_args,
    _explanation
)


@pytest.fixture
def resnet50():
    resnet50_path = cached_path(
        "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/validated/vision/classification/resnet/model/resnet50-v1-7.onnx"
    )
    return resnet50_path


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
    process_custom_script("tests/scripts/pytorch.py", args)
    args.seed = 42

    return args


@pytest.fixture
def args_onnx(args, resnet50):
    args.model = resnet50
    args.seed = 100

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
def data(args_custom, model_shape, cpu_device):
    data = try_preprocess(args_custom, model_shape, device=cpu_device)

    return data


@pytest.fixture
def cpu_device():
    device = get_device(gpu=False)

    return device

@pytest.fixture
def exp_custom(args_custom, model_shape, prediction_func, cpu_device):
    exp = _explanation(args_custom, model_shape, prediction_func, cpu_device, db=None)

    return exp


@pytest.fixture
def exp_onnx(args_onnx, cpu_device):
    prediction_func, model_shape = get_prediction_func_from_args(args_onnx)
    exp = _explanation(args_onnx, model_shape, prediction_func, cpu_device, db=None)

    return exp
