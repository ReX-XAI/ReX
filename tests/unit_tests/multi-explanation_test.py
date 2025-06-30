import logging
import os

import numpy as np
import pytest
from rex_xai.utils._utils import Strategy
from rex_xai.explanation.rex import calculate_responsibility
from rex_xai.explanation.explanation import Explanation
from rex_xai.explanation.multi_explanation import MultiExplanation


@pytest.mark.parametrize("spotlights", [5, 10])
def test_multiexplanation(data_multi, args_multi, prediction_func, spotlights, caplog):
    args_multi.spotlights = spotlights

    maps, run_stats = calculate_responsibility(data_multi, args_multi, prediction_func)

    exp = Explanation(maps, prediction_func, data_multi, args_multi, run_stats)
    args_multi.strategy = Strategy.Global
    exp.extract()

    multi_exp = MultiExplanation(
        maps, prediction_func, data_multi, args_multi, run_stats
    )
    caplog.set_level(logging.INFO)
    args_multi.strategy = Strategy.MultiSpotlight
    multi_exp.extract()

    n_exp = 0
    for record in caplog.records:
        print(record)
        if "found with" in record.message:
            n_exp += 1

    assert (
        caplog.records[-1].message
        == f"ReX has found a total of {n_exp} explanations via spotlight search"
    )
    assert n_exp == len(multi_exp.explanations)
    assert len(multi_exp.explanations) <= spotlights  # always true
    assert (
        len(multi_exp.explanations) == spotlights
    )  # not always true but is for this data/parameters
    assert np.array_equal(
        multi_exp.explanations[0].detach().cpu().numpy(), exp.sufficiency_mask
    )  # first explanation is global explanation


def test_multiexplanation_save_composite(exp_multi, tmp_path):
    clauses = exp_multi.separate_by(0.0)

    p = tmp_path / "exp.png"
    exp_multi.save(path=p, multi_style="composite", clauses=None)

    assert os.path.exists(p)
    assert os.stat(p).st_size > 0

    for c in clauses:
        exp_multi.save(path=p, multi_style="composite", clauses=c)
        clause_path = tmp_path / f"exp_{c}.png"
        assert os.path.exists(clause_path)
        assert os.stat(clause_path).st_size > 0


def test_multiexplanation_save_separate(exp_multi, tmp_path):
    p = tmp_path / "exp.png"
    exp_multi.save(path=p, multi_style="separate")

    for i in range(len(exp_multi.explanations)):
        exp_path = tmp_path / f"exp_{i}.png"
        assert os.path.exists(exp_path)
        assert os.stat(exp_path).st_size > 0
