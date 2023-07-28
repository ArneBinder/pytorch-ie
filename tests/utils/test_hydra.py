from importlib import import_module

import pytest

from pytorch_ie.utils.hydra import InstantiationException, resolve_target


def test_resolve_target_string():
    target_str = "pytorch_ie.utils.hydra.resolve_target"
    target = resolve_target(target_str)
    assert target == resolve_target


def test_resolve_target_not_found():
    with pytest.raises(InstantiationException):
        resolve_target("does.not.exist", full_key="full_key")


def test_resolve_target_empty_path():
    with pytest.raises(InstantiationException):
        resolve_target("")


def test_resolve_target_empty_part():
    with pytest.raises(InstantiationException):
        resolve_target("pie_utils..hydra.resolve_target")


def test_resolve_target_from_src():
    resolve_target("src.pytorch_ie.utils.hydra.resolve_target")


def test_resolve_target_from_src_not_found():
    with pytest.raises(InstantiationException):
        resolve_target("tests.fixtures.not_loadable")


def test_resolve_target_not_loadable(monkeypatch):
    # Normally, import_module will raise ModuleNotFoundError, but we want to test the case
    # in _locate where it raises a different exception.
    # So we mock the import_module function to raise a different exception on the second call
    # (the first call is important to succeed because otherwise we just check the first try/except block).
    class MockImportModule:
        def __init__(self):
            self.counter = 0

        def __call__(self, path):
            if self.counter < 1:
                self.counter += 1
                return import_module(path)
            raise Exception("Custom exception")

    # Apply the monkeypatch to replace import_module with our mock function
    monkeypatch.setattr("importlib.import_module", MockImportModule())

    with pytest.raises(Exception):
        resolve_target("src.invalid_attr")


def test_resolve_target_not_callable_with_full_key():
    with pytest.raises(InstantiationException):
        resolve_target("pie_utils.hydra", full_key="full_key")
