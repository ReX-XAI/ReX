import logging
import os

import numpy as np
import pytest
from rex_xai._utils import Strategy
from rex_xai.explanation import calculate_responsibility, _explanation
from rex_xai.multi_explanation import MultiExplanation
from rex_xai.extraction import Explanation


@pytest.mark.parametrize("spotlights", [5, 10])
def test_multiexplanation(data_multi, args_multi, prediction_func, spotlights, caplog):
    args_multi.spotlights = spotlights

    maps, run_stats = calculate_responsibility(data_multi, args_multi, prediction_func)
    
    exp = Explanation(maps, prediction_func, data_multi, args_multi, run_stats)
    exp.extract(method=Strategy.Global)

    multi_exp = MultiExplanation(
        maps, prediction_func, data_multi, args_multi, run_stats
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


def test_multiexplanation_save_composite(args_multi, model_shape, prediction_func, cpu_device, tmp_path):
    exp = _explanation(args_multi, model_shape, prediction_func, cpu_device, db=None)
    clauses = exp.separate_by(0.0)

    p = tmp_path / "exp.png"
    exp.save(path=p, multi_style="composite", clauses=None)

    assert os.path.exists(p)
    assert os.stat(p).st_size > 0

    exp.save(path=p, multi_style="composite", clauses=clauses)
    for c in clauses:
        clause_path = tmp_path /f"exp_{c}.png"
        assert os.path.exists(clause_path)
        assert os.stat(clause_path).st_size > 0


def test_multiexplanation_save_separate(args_multi, model_shape, prediction_func, cpu_device, tmp_path):
    exp = _explanation(args_multi, model_shape, prediction_func, cpu_device, db=None)

    p = tmp_path / "exp.png"
    exp.save(path=p, multi_style="separate")

    for i in range(len(exp.explanations)):
        exp_path = tmp_path /f"exp_{i}.png"
        assert os.path.exists(exp_path)
        assert os.stat(exp_path).st_size > 0
