from importlib import import_module

import pytest

from pytorch_ie.core import Document
from pytorch_ie.documents import TextBasedDocument
from pytorch_ie.utils.hydra import (
    InstantiationException,
    resolve_optional_document_type,
    resolve_target,
    serialize_document_type,
)
from tests.conftest import TestDocument


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


def test_resolve_optional_document_type():

    assert resolve_optional_document_type(Document) == Document
    assert resolve_optional_document_type("pytorch_ie.core.Document") == Document

    assert resolve_optional_document_type(TextBasedDocument) == TextBasedDocument
    assert (
        resolve_optional_document_type("pytorch_ie.documents.TextBasedDocument")
        == TextBasedDocument
    )


def test_resolve_optional_document_type_none():
    assert resolve_optional_document_type(None) is None


class NoDocument:
    pass


def test_resolve_optional_document_type_no_document():
    with pytest.raises(TypeError) as excinfo:
        resolve_optional_document_type(NoDocument)
    assert (
        str(excinfo.value)
        == "(resolved) document_type must be a subclass of Document, but it is: <class 'tests.utils.test_hydra.NoDocument'>"
    )

    with pytest.raises(TypeError) as excinfo:
        resolve_optional_document_type("tests.utils.test_hydra.NoDocument")
    assert (
        str(excinfo.value)
        == "(resolved) document_type must be a subclass of Document, but it is: <class 'tests.utils.test_hydra.NoDocument'>"
    )


def test_serialize_document_type():
    serialized_dt = serialize_document_type(TestDocument)
    assert serialized_dt == "tests.conftest.TestDocument"
    resolved_dt = resolve_optional_document_type(serialized_dt)
    assert resolved_dt == TestDocument
