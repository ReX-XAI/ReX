import pytest
from rex_xai.input.config import validate_args
from rex_xai.rex import (
    predict_target,
    try_preprocess,
)


def test_preprocess_nii_notimplemented(args, model_shape, cpu_device, caplog):
    args.path = "tests/test_data/dog.nii"
    data = try_preprocess(args, model_shape, device=cpu_device)

    assert data == NotImplemented
    assert caplog.records[0].msg == "we do not (yet) handle nifti files generically"


def test_predict_target(data, prediction_func):
    target = predict_target(data, prediction_func)

    assert target.classification == 207
    assert target.confidence == pytest.approx(0.253237, abs=2.5e-6)


def test_validate_args(args):
    args.path = None  #  type: ignore
    with pytest.raises(FileNotFoundError):
        validate_args(args)


def test_preprocess_rgba(args, model_shape, prediction_func, cpu_device, caplog):
    args.path = "assets/rex_logo.png"
    data = try_preprocess(args, model_shape, device=cpu_device)
    predict_target(data, prediction_func)
    
    assert caplog.records[0].msg == "RGBA input image provided, converting to RGB"
    assert data.mode == "RGB"
    assert data.input.mode == "RGB"
    assert data.data is not None
    assert data.data.shape[1] == 3 # batch, channels, height, width
