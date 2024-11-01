from rex_xai.config import CausalArgs
from rex_xai._utils import get_device
from rex_xai.explanation import try_preprocess

args = CausalArgs()
args.path = "imgs/dog.jpg"
model_shape = ['N', 3, 224, 224] # may not be correct/appropriate, check!
device = get_device(gpu=False)

def test_preprocess(snapshot):
    data = try_preprocess(args, model_shape, device=device)
    assert data == snapshot


def test_preprocess_spectral_mask_on_image_returns_warning(caplog, snapshot):
    args.mask_value = "spectral"
    data = try_preprocess(args, model_shape, device=device)

    assert args.mask_value == 0
    assert caplog.records[0].msg == "spectral is not suitable for images. Changing mask_value to 0"
    assert data == snapshot


def test_preprocess_nii_notimplemented(caplog):
    args.path = "imgs/dog.nii"
    data = try_preprocess(args, model_shape, device=device)

    assert data == NotImplemented
    assert caplog.records[0].msg == "we do not (yet) handle nifti files generically"


def test_preprocess_npy(caplog, snapshot):
    args.path = "imgs/DoublePeakClass 0 Mean.npy"
    try_preprocess(args, model_shape, device=device)

    assert caplog.records[0].msg == "we do not generically handle this datatype"
    
    args.mode = "tabular"
    data = try_preprocess(args, model_shape, device=device)
    assert data == snapshot

