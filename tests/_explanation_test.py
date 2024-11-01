from rex_xai._utils import get_device
from rex_xai.config import CausalArgs, process_custom_script
from rex_xai.explanation import _explanation, get_prediction_func_from_args
from syrupy.extensions.amber.serializer import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type

args = CausalArgs()
args.path = "imgs/dog.jpg"
args.iters = 1
args.search_limit = 1000
args.seed = 42

process_custom_script("tests/scripts/pytorch.py", args)
prediction_func, model_shape = get_prediction_func_from_args(args)
device = get_device(gpu=False)


def test__explanation_snapshot(snapshot):
    exp = _explanation(args, model_shape, prediction_func, device, db=None)
    
    assert exp == snapshot(exclude=props("obj_function", "spotlight_objective_function"), # pointers to functions, will differ between runs
        matcher=path_type(
            types=(CausalArgs,),
            replacer=lambda data, _: AmberDataSerializer.object_as_named_tuple(data), # needed to allow exclude to work for custom classes
        ),)

