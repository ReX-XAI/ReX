import pytest
from rex_xai._utils import ReXTomlError
from rex_xai.config import CausalArgs, process_config_dict, validate_args


def test_validate_args(args):
    args.path = None  #  type: ignore
    with pytest.raises(FileNotFoundError):
        validate_args(args)


def test_process_config_dict_blend_invalid(caplog):
    args = CausalArgs()
    config_dict = {"causal": {"distribution": {"blend": 20}}}
    process_config_dict(config_dict, args)
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert (
            caplog.records[0].message
            == "Invalid value '20': must be between 0.0 and 1.0"
        )


def test_process_config_dict_permitted_overlap_invalid(caplog):
    args = CausalArgs()
    config_dict = {"explanation": {"multi": {"permitted_overlap": -5}}}
    process_config_dict(config_dict, args)
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert (
            caplog.records[0].message
            == "Invalid value '-5': must be between 0.0 and 1.0"
        )


def test_process_config_dict_iters_invalid(caplog):
    args = CausalArgs()
    config_dict = {"causal": {"iters": 0}}
    process_config_dict(config_dict, args)
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert caplog.records[0].message == "Invalid value '0': must be more than 0.0"


def test_process_config_dict_raw_invalid(caplog):
    args = CausalArgs()
    config_dict = {"rex": {"visual": {"raw": 100}}}
    process_config_dict(config_dict, args)
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert caplog.records[0].message == "Invalid value '100': must be boolean"


def test_process_config_dict_multi_style_invalid(caplog):
    args = CausalArgs()
    config_dict = {"rex": {"visual": {"multi_style": "an-invalid-style"}}}
    process_config_dict(config_dict, args)
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert (
            caplog.records[0].message
            == "Invalid value 'an-invalid-style' for multi_style, must be 'composite' or 'separate'"
        )
