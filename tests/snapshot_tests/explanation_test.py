import pytest
from rex_xai.mutants.distributions import Distribution
from rex_xai.explanation.rex import calculate_responsibility, predict_target


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
    maps, _ = calculate_responsibility(data_custom, args_custom, prediction_func)
    target_map = maps.get(data_custom.target.classification)

    assert hash(tuple(target_map.reshape(-1))) == snapshot
