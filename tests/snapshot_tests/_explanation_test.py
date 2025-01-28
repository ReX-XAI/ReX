import pytest
from rex_xai.config import CausalArgs, Strategy
from rex_xai.explanation import _explanation, analyze
from syrupy.extensions.amber.serializer import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type


@pytest.mark.parametrize("batch", [1, 64])
def test__explanation_snapshot(
    args_custom, model_shape, prediction_func, cpu_device, batch, snapshot
):
    args_custom.batch = batch
    exp = _explanation(args_custom, model_shape, prediction_func, cpu_device, db=None)

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


@pytest.mark.parametrize("batch", [1, 64])
def test__explanation_snapshot_diff_model_shape(
    args_torch_swin_v2_t,
    model_shape_swin_v2_t,
    prediction_func_swin_v2_t,
    cpu_device,
    batch,
    snapshot,
):
    args_torch_swin_v2_t.batch = batch

    exp = _explanation(
        args_torch_swin_v2_t,
        model_shape_swin_v2_t,
        prediction_func_swin_v2_t,
        cpu_device,
        db=None,
    )

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


@pytest.mark.parametrize("strategy", [Strategy.Global, Strategy.Spatial])
def test_extract_analyze(exp_custom, strategy, snapshot):
    exp_custom.extract(strategy)
    results = analyze(exp_custom, "RGB")
    results_rounded = {k: round(v, 4) for k, v in results.items()}

    assert exp_custom.final_mask == snapshot
    assert results_rounded == snapshot
