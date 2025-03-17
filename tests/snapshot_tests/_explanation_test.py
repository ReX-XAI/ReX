import pytest
from rex_xai.explanation import _explanation, analyze
from rex_xai._utils import Strategy

@pytest.mark.parametrize("batch_size", [1, 64])
def test__explanation_snapshot(
    args_custom, model_shape, prediction_func, cpu_device, batch_size, snapshot_explanation
):
    args_custom.batch_size = batch_size
    exp = _explanation(args_custom, model_shape, prediction_func, cpu_device, db=None)

    assert exp == snapshot_explanation
    assert hash(tuple(exp.explanation.reshape(-1).tolist())) == snapshot_explanation


@pytest.mark.parametrize("batch_size", [1, 64])
def test__explanation_snapshot_diff_model_shape(
    args_torch_swin_v2_t,
    model_shape_swin_v2_t,
    prediction_func_swin_v2_t,
    cpu_device,
    batch_size,
    snapshot_explanation
):
    args_torch_swin_v2_t.batch_size = batch_size

    exp = _explanation(
        args_torch_swin_v2_t,
        model_shape_swin_v2_t,
        prediction_func_swin_v2_t,
        cpu_device,
        db=None,
    )

    assert exp == snapshot_explanation
    assert hash(tuple(exp.explanation.reshape(-1).tolist())) == snapshot_explanation


@pytest.mark.parametrize("strategy", [Strategy.Global, Strategy.Spatial])
def test_extract_analyze(exp_custom, strategy, snapshot):
    exp_custom.extract(strategy)
    results = analyze(exp_custom, "RGB")
    results_rounded = {k: round(v, 4) for k, v in results.items() if v is not None}

    assert hash(tuple(exp_custom.final_mask.reshape(-1).tolist())) == snapshot
    assert results_rounded == snapshot


def test_multiexplanation_snapshot(
    args_multi, model_shape, prediction_func, cpu_device, snapshot_explanation
):
    exp = _explanation(args_multi, model_shape, prediction_func, cpu_device, db=None)
    clauses = exp.separate_by(0.0)

    assert exp == snapshot_explanation
    assert clauses == snapshot_explanation
    for explanation in exp.explanations:
        assert hash(tuple(explanation.reshape(-1).tolist())) == snapshot_explanation
