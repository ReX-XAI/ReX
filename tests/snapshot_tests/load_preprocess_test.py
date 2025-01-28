from rex_xai.explanation import (
    load_and_preprocess_data,
    try_preprocess,
)


def test_preprocess(args, model_shape, cpu_device, snapshot):
    data = load_and_preprocess_data(model_shape, cpu_device, args)
    assert data == snapshot


def test_preprocess_custom(args_custom, model_shape, cpu_device, snapshot):
    data = load_and_preprocess_data(model_shape, cpu_device, args_custom)
    assert data == snapshot


def test_preprocess_spectral_mask_on_image_returns_warning(
    args, model_shape, cpu_device, snapshot, caplog
):
    args.mask_value = "spectral"
    data = try_preprocess(args, model_shape, device=cpu_device)

    assert args.mask_value == 0
    assert (
        caplog.records[0].msg
        == "spectral is not suitable for images. Changing mask_value to 0"
    )
    assert data == snapshot


def test_preprocess_npy(args, model_shape, cpu_device, snapshot, caplog):
    args.path = "tests/test_data/DoublePeakClass 0 Mean.npy"
    try_preprocess(args, model_shape, device=cpu_device)

    assert caplog.records[0].msg == "we do not generically handle this datatype"

    args.mode = "tabular"
    data = try_preprocess(args, model_shape, device=cpu_device)
    assert data == snapshot
