import pytest

from rex_xai._utils import get_device
from rex_xai.config import CausalArgs, process_custom_script
from rex_xai.explanation import _explanation, get_prediction_func_from_args
from syrupy.extensions.amber.serializer import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type


@pytest.fixture
def args_custom_script():
    args = CausalArgs()
    args.path = "imgs/dog.jpg"
    args.iters = 2
    args.search_limit = 1000
    args.seed = 42
    args.gpu = False
    process_custom_script("tests/scripts/pytorch.py", args)
    return args


device = get_device(gpu=False)


def test__explanation_snapshot(args_custom_script, snapshot):
    prediction_func, model_shape = get_prediction_func_from_args(args_custom_script)
    exp = _explanation(
        args_custom_script, model_shape, prediction_func, device, db=None
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
