import pytest
from rex_xai._utils import ReXTomlError
from rex_xai.config import CausalArgs, validate_args


def test_no_path(args):
    args.path = None  #  type: ignore
    with pytest.raises(FileNotFoundError):
        validate_args(args)


def test_blend_invalid(caplog):
    args = CausalArgs()
    args.blend = 20
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert (
            caplog.records[0].message
            == "Invalid value '20': must be between 0.0 and 1.0"
        )


def test_permitted_overlap_invalid(caplog):
    args = CausalArgs()
    args.permitted_overlap = -5
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert (
            caplog.records[0].message
            == "Invalid value '-5': must be between 0.0 and 1.0"
        )


def test_iters_invalid(caplog):
    args = CausalArgs()
    args.iters = 0
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert caplog.records[0].message == "Invalid value '0': must be more than 0.0"


def test_raw_invalid(caplog):
    args = CausalArgs()
    args.raw = 100  # type: ignore
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert caplog.records[0].message == "Invalid value '100': must be boolean"


def test_multi_style_invalid(caplog):
    args = CausalArgs()
    args.multi_style = "an-invalid-style"
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert (
            caplog.records[0].message
            == "Invalid value 'an-invalid-style' for multi_style, must be 'composite' or 'separate'"
        )


def test_queue_len_invalid(caplog):
    args = CausalArgs()
    args.queue_len = 7.5  # type: ignore
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert (
            caplog.records[0].message
            == "Invalid value '7.5' for queue_len, must be 'all' or an integer"
        )


def test_distribution_args_invalid(caplog):
    args = CausalArgs()
    args.distribution_args = 1  # type: ignore
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert caplog.records[0].message == "distribution args must be length 2, not 1"

    args.distribution_args = [0, -1]
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert (
            caplog.records[0].message
            == "All values in distribution args must be more than zero"
        )


def test_colour_map_invalid(caplog):
    args = CausalArgs()
    args.heatmap_colours = "RedBlue"
    with pytest.raises(ReXTomlError):
        validate_args(args)
        assert (
            caplog.records[0].message
            == "Invalid colourmap 'RedBlue', must be a valid matplotlib colourmap"
        )
