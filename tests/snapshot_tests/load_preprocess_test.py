import pytest
from rex_xai.utils._utils import ReXDataError
from rex_xai.rex import (
    load_and_preprocess_data,
    try_preprocess,
    get_prediction_func_from_args
)


def test_preprocess(args, model_shape, cpu_device, snapshot):
    data = load_and_preprocess_data(model_shape, cpu_device, args)
    assert data == snapshot


def test_preprocess_custom(args_custom, model_shape, cpu_device, snapshot):
    data = load_and_preprocess_data(model_shape, cpu_device, args_custom)
    assert data == snapshot


def test_preprocess_spectral_mask_on_image_returns_warning(
    args, model_shape, cpu_device, snapshot, caplog
):
    args.mask_value = "spectral"
    data = try_preprocess(args, model_shape, device=cpu_device)

    assert args.mask_value == 0
    assert (
        caplog.records[0].msg
        == "spectral is not suitable for images. Changing mask_value to 0"
    )
    assert data == snapshot


def test_preprocess_npy(args, DNA_model, cpu_device, snapshot, caplog):
    args.path = "tests/test_data/DoublePeakClass 0 Mean.npy"
    args.model = DNA_model
    _, model_shape = get_prediction_func_from_args(args)

    try_preprocess(args, model_shape, device=cpu_device)

    assert caplog.records[0].msg == "we do not generically handle this datatype"

    args.mode = "tabular"
    with pytest.raises(ReXDataError):
        try_preprocess(args, model_shape, device=cpu_device)

def test_preprocess_incompatible_shapes(args, model_shape, cpu_device, caplog):
    args.path = "tests/test_data/DoublePeakClass 0 Mean.npy"
    args.mode = "tabular"

    with pytest.raises(ReXDataError):
        try_preprocess(args, model_shape, device=cpu_device)

