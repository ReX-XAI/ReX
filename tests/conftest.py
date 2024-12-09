import pytest

from cached_path import cached_path

@pytest.fixture
def resnet50():
    resnet50_path = cached_path("https://github.com/onnx/models/blob/bec48b6a70e5e9042c0badbaafefe4454e072d08/validated/vision/classification/resnet/model/resnet50-v1-7.onnx")
    return resnet50_path