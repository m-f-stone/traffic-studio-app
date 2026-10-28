"""Device resolution must degrade to CPU instead of raising when CUDA is absent."""
from studio.devices import resolve_device


def test_resolve_device_without_cuda():
    assert resolve_device(None) == ("cpu", False)
    assert resolve_device("auto") == ("cpu", False)
    assert resolve_device("cpu") == ("cpu", False)
    assert resolve_device("0") == ("cpu", False)
    assert resolve_device(0) == ("cpu", False)
