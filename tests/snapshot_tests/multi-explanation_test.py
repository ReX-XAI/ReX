from rex_xai.explanation import _explanation


def test_multiexplanation_snapshot(
    args_multi, model_shape, prediction_func, cpu_device, snapshot_explanation
):
    exp = _explanation(args_multi, model_shape, prediction_func, cpu_device, db=None)
    clauses = exp.separate_by(0.0)

    assert exp == snapshot_explanation
    assert clauses == snapshot_explanation
    assert hash(tuple(exp.explanation.reshape(-1).tolist())) == snapshot_explanation
