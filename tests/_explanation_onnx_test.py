import pytest

from rex_xai.config import CausalArgs, Strategy
from rex_xai.explanation import _explanation, get_prediction_func_from_args
from syrupy.extensions.amber.serializer import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type


def test__explanation_snapshot(args_onnx, cpu_device, snapshot):
    prediction_func, model_shape = get_prediction_func_from_args(args_onnx)
    exp = _explanation(args_onnx, model_shape, prediction_func, cpu_device, db=None)

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


@pytest.mark.parametrize("strategy", ["global", "spatial"])
def test_extract(exp_onnx, strategy, snapshot):
    if strategy == "global":
        strategy = Strategy.Global
    elif strategy == "spatial":
        strategy = Strategy.Spatial
    else:
        raise ValueError("invalid strategy!")
    exp_onnx.extract(strategy)
    
    assert(exp_onnx.final_mask == snapshot)