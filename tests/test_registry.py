import pytest

from homura.register import Registry


def test_registry():
    MODEL_REGISTRY = Registry('model')
    MODEL_REGISTRY2 = Registry('model')
    assert MODEL_REGISTRY is MODEL_REGISTRY2

    @MODEL_REGISTRY.register
    def something():
        return 1

    @MODEL_REGISTRY.register
    def anything():
        return 2

    assert MODEL_REGISTRY('something')() == 1

    with pytest.raises(KeyError):
        @MODEL_REGISTRY.register
        def something():
            pass
