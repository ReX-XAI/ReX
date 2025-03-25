import pytest
from cached_path import cached_path
from rex_xai.config import Strategy
from rex_xai.rex import _explanation, get_prediction_func_from_args


@pytest.fixture(scope="session")
def DNA_model():
    DNA_model_path = cached_path(
        "https://github.com/ReX-XAI/models/raw/6f66a5c0e1480411436be828ee8312e72f0035e1/spectral/simple_DNA_model.onnx"
    )
    return DNA_model_path


@pytest.fixture
def args_spectral(args, DNA_model):
    args.model = DNA_model
    args.path = "tests/test_data/spectrum_class_DNA.npy"
    args.mode = "spectral"
    args.mask_value = "spectral"
    args.seed = 15
    args.strategy = Strategy.Global

    return args


@pytest.mark.parametrize(
    "batch_size", [1, 64]
)
def test__explanation_snapshot(args_spectral, cpu_device, batch_size, snapshot_explanation):
    args_spectral.batch_size = batch_size
    prediction_func, model_shape = get_prediction_func_from_args(args_spectral)
    exp = _explanation(args_spectral, model_shape, prediction_func, cpu_device, db=None)

    assert exp == snapshot_explanation
    assert hash(tuple(exp.explanation.reshape(-1).tolist())) == snapshot_explanation
