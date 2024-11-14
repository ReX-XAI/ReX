from rex_xai._utils import get_device
from rex_xai.config import CausalArgs
from rex_xai.explanation import _explanation, get_prediction_func_from_args
from syrupy.extensions.amber.serializer import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type
from cached_path import cached_path

args = CausalArgs()
args.path = "imgs/dog.jpg"
args.iters = 1
args.search_limit = 1000
args.seed = 42

model_path = cached_path("https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx")
args.model = model_path

prediction_func, model_shape = get_prediction_func_from_args(args)
device = get_device(gpu=False)


def test__explanation_snapshot(snapshot):
    exp = _explanation(args, model_shape, prediction_func, device, db=None)

    assert exp == snapshot(
        exclude=props(
            "obj_function", "spotlight_objective_function"
        ),  # pointers to functions, will differ between runs
        matcher=path_type(
            types=(CausalArgs,),
            replacer=lambda data, _: AmberDataSerializer.object_as_named_tuple(
                data
            ),  # needed to allow exclude to work for custom classes
        ),
    )
