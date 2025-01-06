from rex_xai.config import CausalArgs
from rex_xai.explanation import _explanation
from syrupy.extensions.amber.serializer import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type


def test__explanation_snapshot(
    args_custom, model_shape, prediction_func, cpu_device, snapshot
):
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
