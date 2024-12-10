import pytest

from rex_xai._utils import get_device
from rex_xai.config import CausalArgs
from rex_xai.explanation import _explanation, get_prediction_func_from_args
from syrupy.extensions.amber.serializer import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type

@pytest.fixture
def args_onnx(resnet50):
    args = CausalArgs()
    args.path = "imgs/dog.jpg"
    args.iters = 1
    args.search_limit = 1000
    args.seed = 100
    args.model = resnet50
    return args

device = get_device(gpu=False)


def test__explanation_snapshot(args_onnx, snapshot):

    prediction_func, model_shape = get_prediction_func_from_args(args_onnx)
    exp = _explanation(args_onnx, model_shape, prediction_func, device, db=None)

    assert exp == snapshot(
        exclude=props(
            "obj_function", "spotlight_objective_function", "model"
        ),  # paths that differ between systems, pointers to functions that will differ between runs
        matcher=path_type(
            types=(CausalArgs,),
            replacer=lambda data, _: AmberDataSerializer.object_as_named_tuple(
                data
            ),  # needed to allow exclude to work for custom classes
        ),
    )
