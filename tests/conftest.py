import pytest

from cached_path import cached_path

from rex_xai.config import CausalArgs, process_custom_script
from rex_xai._utils import get_device
from rex_xai.explanation import (
    try_preprocess,
    get_prediction_func_from_args,
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
    process_custom_script("tests/scripts/pytorch.py", args)

    return args


@pytest.fixture
def model_shape(args):
    prediction_func, model_shape = get_prediction_func_from_args(args)

    return model_shape


@pytest.fixture
def prediction_func(args):
    prediction_func, model_shape = get_prediction_func_from_args(args)

    return prediction_func


@pytest.fixture
def data(args, model_shape, cpu_device):
    data = try_preprocess(args, model_shape, device=cpu_device)

    return data


@pytest.fixture
def cpu_device():
    device = get_device(gpu=False)

    return device
