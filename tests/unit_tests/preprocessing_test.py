import pytest
from rex_xai.explanation import (
    predict_target,
    try_preprocess,
)
from rex_xai.config import validate_args

def test_preprocess_nii_notimplemented(args, model_shape, cpu_device, caplog):
    args.path = "tests/test_data/dog.nii"
    data = try_preprocess(args, model_shape, device=cpu_device)

    assert data == NotImplemented
    assert caplog.records[0].msg == "we do not (yet) handle nifti files generically"


def test_predict_target(data, prediction_func):
    target = predict_target(data, prediction_func)

    assert target.classification == 207
    assert target.confidence == pytest.approx(0.253237, abs=2.5e-6)

