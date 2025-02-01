import pytest
from rex_xai.config import Strategy
from rex_xai.explanation import _explanation, analyze

@pytest.mark.parametrize("batch", [1, 64])
def test__explanation_snapshot(
    args_custom, model_shape, prediction_func, cpu_device, batch, snapshot_explanation
):
    args_custom.batch = batch
    exp = _explanation(args_custom, model_shape, prediction_func, cpu_device, db=None)

    assert exp == snapshot_explanation
    assert hash(tuple(exp.explanation.reshape(-1).tolist())) == snapshot_explanation


@pytest.mark.parametrize("batch", [1, 64])
def test__explanation_snapshot_diff_model_shape(
    args_torch_swin_v2_t,
    model_shape_swin_v2_t,
    prediction_func_swin_v2_t,
    cpu_device,
    batch,
    snapshot_explanation
):
    args_torch_swin_v2_t.batch = batch

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
    results_rounded = {k: round(v, 4) for k, v in results.items()}

    assert hash(tuple(exp_custom.final_mask.reshape(-1).tolist())) == snapshot
    assert results_rounded == snapshot
