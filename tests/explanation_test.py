import pytest
import copy

from rex_xai.explanation import (
    try_preprocess,
    predict_target,
    load_and_preprocess_data,
    validate_args,
    calculate_responsibility
)
from rex_xai.distributions import Distribution
from rex_xai.config import CausalArgs
from syrupy.extensions.amber.serializer import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type


def test_preprocess(args, model_shape, cpu_device, snapshot):
    data = load_and_preprocess_data(model_shape, cpu_device, args)
    assert data == snapshot


def test_preprocess_custom(args_custom, model_shape, cpu_device, snapshot):
    data = load_and_preprocess_data(model_shape, cpu_device, args_custom)
    assert data == snapshot


def test_preprocess_spectral_mask_on_image_returns_warning(
    args, model_shape, cpu_device, snapshot, caplog
):
    args = copy.deepcopy(args)
    args.mask_value = "spectral"
    data = try_preprocess(args, model_shape, device=cpu_device)

    assert args.mask_value == 0
    assert (
        caplog.records[0].msg
        == "spectral is not suitable for images. Changing mask_value to 0"
    )
    assert data == snapshot


def test_preprocess_nii_notimplemented(args, model_shape, cpu_device, caplog):
    args = copy.deepcopy(args)
    args.path = "tests/test_data/dog.nii"
    data = try_preprocess(args, model_shape, device=cpu_device)

    assert data == NotImplemented
    assert caplog.records[0].msg == "we do not (yet) handle nifti files generically"


def test_preprocess_npy(args, model_shape, cpu_device, snapshot, caplog):
    args = copy.deepcopy(args)
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
    args = copy.deepcopy(args)
    args.path = None  #  type: ignore
    with pytest.raises(FileNotFoundError):
        validate_args(args)


@pytest.mark.parametrize("distribution,dist_args", [("uniform", []),
                                                    ("betabinom", [1, 1]), ("betabinom", [0.5, 0.5]),
                                                    ("betabinom", [1, 0.5]), ("betabinom", [0.5, 1])])
def test_calculate_responsibility(data_custom, args_custom, prediction_func, distribution, dist_args, snapshot):
    args_custom = copy.deepcopy(args_custom)
    data_custom = copy.deepcopy(data_custom)
    if distribution == "uniform":
        args_custom.distribution = Distribution.Uniform
    if distribution == "betabinom":
        args_custom.distribution = Distribution.BetaBinomial
        args_custom.distribution_args = dist_args
    data_custom.target = predict_target(data_custom, prediction_func)
    exp = calculate_responsibility(data_custom, args_custom, prediction_func)

    assert (
        exp
        == snapshot(
            exclude=props(
                "obj_function",
                "spotlight_objective_function",
                "custom",
                "custom_location",
            ),  # paths that differ between systems, pointers to functions that will differ between runs
            matcher=path_type(
                types=(CausalArgs,),
                replacer=lambda data, _: AmberDataSerializer.object_as_named_tuple(
                    data
                ),  # needed to allow exclude to work for custom classes
            ),
        )
    )
