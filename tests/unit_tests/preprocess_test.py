import pytest

from rex_xai.explanation import (
    try_preprocess,
    predict_target,
    load_and_preprocess_data,
    validate_args
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


def test_preprocess_nii_notimplemented(args, model_shape, cpu_device, caplog):
    args.path = "tests/test_data/dog.nii"
    data = try_preprocess(args, model_shape, device=cpu_device)

    assert data == NotImplemented
    assert caplog.records[0].msg == "we do not (yet) handle nifti files generically"


def test_preprocess_npy(args, model_shape, cpu_device, snapshot, caplog):
    args.path = "tests/test_data/DoublePeakClass 0 Mean.npy"
    try_preprocess(args, model_shape, device=cpu_device)

    assert caplog.records[0].msg == "we do not generically handle this datatype"

    args.mode = "tabular"
    data = try_preprocess(args, model_shape, device=cpu_device)
    assert data == snapshot


def test_predict_target(data, prediction_func):
    target = predict_target(data, prediction_func)

    assert target.classification == 207
    assert target.confidence == pytest.approx(0.253237, abs=2.5e-6)


def test_validate_args(args):
    args.path = None  #  type: ignore
    with pytest.raises(FileNotFoundError):
        validate_args(args)
