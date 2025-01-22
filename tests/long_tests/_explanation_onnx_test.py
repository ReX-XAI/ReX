import pytest
from rex_xai.config import CausalArgs, Strategy
from rex_xai.explanation import _explanation, analyze, get_prediction_func_from_args
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


@pytest.mark.parametrize("strategy", [Strategy.Global, Strategy.Spatial])
def test_extract_analyze(exp_onnx, strategy, snapshot):
    exp_onnx.extract(strategy)
    results = analyze(exp_onnx, "RGB")
    results_rounded = {k: round(v, 4) for k, v in results.items()}

    assert exp_onnx.final_mask == snapshot
    assert results_rounded == snapshot
