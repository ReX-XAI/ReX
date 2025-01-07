import pytest
from cached_path import cached_path
from rex_xai._utils import get_device
from rex_xai.config import CausalArgs, process_custom_script, Strategy
from rex_xai.explanation import (
    calculate_responsibility,
    get_prediction_func_from_args,
    load_and_preprocess_data,
    predict_target,
    try_preprocess,
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
def data_custom(args_custom, model_shape, cpu_device):
    data = load_and_preprocess_data(model_shape, cpu_device, args_custom)
    data.set_mask_value(args_custom.mask_value, device=data.device)
    return data


@pytest.fixture(scope="session")
def cpu_device():
    device = get_device(gpu=False)

    return device


@pytest.fixture
def exp_custom(data_custom, args_custom, prediction_func):
    data_custom.target = predict_target(data_custom, prediction_func)
    exp = calculate_responsibility(data_custom, args_custom, prediction_func)

    return exp


@pytest.fixture
def exp_onnx(args_onnx, cpu_device):
    prediction_func, model_shape = get_prediction_func_from_args(args_onnx)
    data = load_and_preprocess_data(model_shape, cpu_device, args_onnx)
    data.set_mask_value(args_onnx.mask_value, device=data.device)
    data.target = predict_target(data, prediction_func)
    exp = calculate_responsibility(data, args_onnx, prediction_func)

    return exp

@pytest.fixture
def exp_extracted(exp_custom):
    exp_custom.extract(Strategy.Global)

    return exp_custom