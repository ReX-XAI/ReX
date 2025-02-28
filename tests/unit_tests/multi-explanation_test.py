import logging

import numpy as np
import pytest
from rex_xai._utils import Strategy
from rex_xai.explanation import calculate_responsibility
from rex_xai.multi_explanation import MultiExplanation


@pytest.mark.parametrize("spotlights", [5, 10])
def test_multiexplanation(data_multi, args_multi, prediction_func, spotlights, caplog):
    args_multi.spotlights = spotlights

    exp = calculate_responsibility(data_multi, args_multi, prediction_func)
    exp.extract(method=Strategy.Global)

    multi_exp = MultiExplanation(
        exp.maps, prediction_func, data_multi, args_multi, exp.run_stats
    )
    caplog.set_level(logging.INFO)
    multi_exp.extract(Strategy.MultiSpotlight)

    n_exp = 0
    for record in caplog.records:
        print(record)
        if "found an explanation" in record.message:
            n_exp += 1
    
    assert (
        caplog.records[-1].message
        == f"ReX has found a total of {n_exp} explanations via spotlight search"
    )
    assert n_exp == len(multi_exp.explanations)
    assert len(multi_exp.explanations) <= spotlights  # always true
    assert (len(multi_exp.explanations) == spotlights)  # not always true but is for this data/parameters
    assert np.array_equal(
        multi_exp.explanations[0].detach().cpu().numpy(), exp.final_mask
    )  # first explanation is global explanation
