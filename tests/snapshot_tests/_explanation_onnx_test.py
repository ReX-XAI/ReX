import pytest
from rex_xai.input.config import Strategy
from rex_xai.rex import _explanation, analyze, get_prediction_func_from_args


def test__explanation_snapshot(args_onnx, cpu_device, snapshot_explanation):
    prediction_func, model_shape = get_prediction_func_from_args(args_onnx)
    exp = _explanation(args_onnx, model_shape, prediction_func, cpu_device, db=None)

    assert exp == snapshot_explanation
    assert hash(tuple(exp.explanation.reshape(-1).tolist())) == snapshot_explanation


@pytest.mark.parametrize("strategy", [Strategy.Global, Strategy.Spatial])
def test_extract_analyze(exp_onnx, strategy, snapshot):
    exp_onnx.extract(strategy)
    results = analyze(exp_onnx, "RGB")
    results_rounded = {k: round(v, 4) for k, v in results.items() if v is not None}

    assert hash(tuple(exp_onnx.final_mask.reshape(-1).tolist())) == snapshot
    assert results_rounded == snapshot
