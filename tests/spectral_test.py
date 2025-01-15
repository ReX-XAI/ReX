import pytest

from rex_xai.config import CausalArgs, Strategy
from rex_xai.explanation import _explanation, get_prediction_func_from_args
from syrupy.extensions.amber.serializer import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type


@pytest.fixture
def args_spectral(args, resnet50):
    args.model = "../models/spectral_testing/simple_DNA_model.onnx"
    args.path = "tests/test_data/spectrum_class_DNA.npy"
    args.mode = 'spectral'
    args.mask_value = 'spectral'
    args.seed = 15
    args.strategy = Strategy.Global

    return args


@pytest.mark.parametrize("batch", [1, 
                                   pytest.param(64, marks=pytest.mark.xfail(reason="ONNXRuntimeError"))])
def test__explanation_snapshot(args_spectral, cpu_device, batch, snapshot):
    args_spectral.batch = batch
    prediction_func, model_shape = get_prediction_func_from_args(args_spectral)
    exp = _explanation(args_spectral, model_shape, prediction_func, cpu_device, db=None)

    assert (
        exp
        == snapshot(
            exclude=props(
                "obj_function", "spotlight_objective_function", "model"
            ),  # paths that differ between systems, pointers to functions that will differ between runs
            matcher=path_type(
                types=(CausalArgs,),
                replacer=lambda data, _: AmberDataSerializer.object_as_named_tuple(
                    data
                ),  # needed to allow exclude to work for custom classes
            ),
        )
    )
