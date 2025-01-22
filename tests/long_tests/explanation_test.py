import pytest
from rex_xai.config import CausalArgs
from rex_xai.distributions import Distribution
from rex_xai.explanation import calculate_responsibility, predict_target
from syrupy.extensions.amber.serializer import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type


@pytest.mark.parametrize(
    "distribution,dist_args",
    [
        (Distribution.Uniform, []),
        (Distribution.BetaBinomial, [1, 1]),
        (Distribution.BetaBinomial, [0.5, 0.5]),
        (Distribution.BetaBinomial, [1, 0.5]),
        (Distribution.BetaBinomial, [0.5, 1]),
    ],
)
def test_calculate_responsibility(
    data_custom, args_custom, prediction_func, distribution, dist_args, snapshot
):
    args_custom.distribution = distribution
    if dist_args:
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
