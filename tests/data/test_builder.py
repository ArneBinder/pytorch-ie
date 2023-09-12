import re
import tempfile
from dataclasses import dataclass
from typing import Type

import pytest
from datasets import DatasetBuilder, Version
from datasets.load import dataset_module_factory, import_main_class

from pytorch_ie.annotations import LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.data.builder import PieDatasetBuilder
from pytorch_ie.documents import TextBasedDocument, TextDocumentWithEntities
from tests import FIXTURES_ROOT

DATASETS_ROOT = FIXTURES_ROOT / "builder" / "datasets"


def test_builder_class():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "single_config"))
    builder_cls = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        builder = builder_cls(cache_dir=tmp_cache_dir)
        assert isinstance(builder, DatasetBuilder)


def test_builder_class_with_kwargs():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "single_config"))
    builder_cls = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        builder = builder_cls(cache_dir=tmp_cache_dir, parameter="test")
        assert isinstance(builder, DatasetBuilder)
        assert builder.config.parameter == "test"


def test_builder_class_with_kwargs_wrong_parameter():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "single_config"))
    builder_cls = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        # this should raise an exception because the base config does not know the parameter
        with pytest.raises(
            TypeError,
            match=re.escape("__init__() got an unexpected keyword argument 'unknown_parameter'"),
        ):
            builder = builder_cls(
                cache_dir=tmp_cache_dir, parameter="test", unknown_parameter="test_unknown"
            )


def test_builder_class_with_base_dataset_kwargs():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "single_config"))
    builder_cls = import_main_class(dataset_module.module_path)
    base_dataset_kwargs = dict(version=Version("0.0.0"), description="new description")
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        builder = builder_cls(cache_dir=tmp_cache_dir, base_dataset_kwargs=base_dataset_kwargs)
        assert isinstance(builder, DatasetBuilder)
        assert builder.base_builder.config.version == "0.0.0"
        assert builder.base_builder.config.description == "new description"


def test_builder_class_with_base_dataset_kwargs_wrong_parameter():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "single_config"))
    builder_cls = import_main_class(dataset_module.module_path)
    base_dataset_kwargs = dict(unknown_base_parameter="base_parameter_value")
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        # this should raise an exception because the base config does not know the parameter
        with pytest.raises(
            TypeError,
            match=re.escape(
                "__init__() got an unexpected keyword argument 'unknown_base_parameter'"
            ),
        ):
            builder = builder_cls(cache_dir=tmp_cache_dir, base_dataset_kwargs=base_dataset_kwargs)


def test_builder_class_multi_configs():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "multi_config"))
    builder_cls = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        with pytest.raises(ValueError, match="Config name is missing."):
            builder = builder_cls(cache_dir=tmp_cache_dir)

        builder = builder_cls(config_name="es", cache_dir=tmp_cache_dir)
        assert isinstance(builder, DatasetBuilder)


def test_builder_class_name_mapping():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "name_mapping"))
    builder_cls = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        builder = builder_cls(config_name="es", cache_dir=tmp_cache_dir)
        assert builder.info.config_name == "es"
        assert builder.base_builder.info.config_name == "nl"

        builder = builder_cls(config_name="nl", cache_dir=tmp_cache_dir)
        assert builder.info.config_name == "nl"
        assert builder.base_builder.info.config_name == "nl"


def test_builder_class_name_mapping_disabled():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "name_mapping_disabled"))
    builder_cls = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        # this should raise an exception because the config name is not passed
        with pytest.raises(ValueError, match="Config name is missing."):
            builder = builder_cls(config_name="es", cache_dir=tmp_cache_dir)

        # here we set the base config name via base_dataset_kwargs
        builder = builder_cls(
            config_name="es", cache_dir=tmp_cache_dir, base_dataset_kwargs=dict(name="nl")
        )
        assert builder.info.config_name == "es"
        assert builder.base_builder.info.config_name == "nl"


def test_builder_class_name_mapping_and_defaults():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "default_config_kwargs"))
    builder_cls = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        # this comes from passing the config as base config name
        builder = builder_cls(config_name="es", cache_dir=tmp_cache_dir)
        assert builder.info.config_name == "es"
        assert builder.base_builder.info.config_name == "es"

        # this gets created by the default setting from BASE_CONFIG_KWARGS_DICT
        builder = builder_cls(config_name="nl", cache_dir=tmp_cache_dir)
        assert builder.info.config_name == "nl"
        assert builder.base_builder.info.config_name == "default"
        assert builder.base_builder.info.version == "0.0.0"


def test_wrong_builder_class_config():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "wrong_builder_class_config"))
    builder_cls = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        # This should raise an exception because the base builder is derived from GeneratorBasedBuilder,
        # but the PIE dataset builder is derived from ArrowBasedBuilder.
        with pytest.raises(
            TypeError,
            match=re.escape(
                "The PyTorch-IE dataset builder class 'Example' is derived from "
                "<class 'datasets.builder.ArrowBasedBuilder'>, but the base builder is not which is not allowed. "
                "The base builder is of type 'Conll2003' that is derived from "
                "<class 'datasets.builder.GeneratorBasedBuilder'>. Consider to derive your PyTorch-IE dataset builder "
                "'Example' from a PyTorch-IE variant of 'GeneratorBasedBuilder'."
            ),
        ):
            builder_cls(cache_dir=tmp_cache_dir)


def test_builder_with_document_converters_rename():
    @dataclass
    class RenamedExampleDocument(TextBasedDocument):
        spans: AnnotationList[LabeledSpan] = annotation_field(target="text")

    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "single_config"))
    builder_cls: Type[PieDatasetBuilder] = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        builder = builder_cls(
            cache_dir=tmp_cache_dir,
            document_converters={
                RenamedExampleDocument: {"entities": "spans"},
            },
        )
    assert isinstance(builder, PieDatasetBuilder)
    assert builder.document_converters == {
        RenamedExampleDocument: {"entities": "spans"},
    }


@dataclass
class ExampleDocumentWithSimpleSpans(TextBasedDocument):
    spans: AnnotationList[Span] = annotation_field(target="text")


def convert_example_document_to_example_document_with_simple_spans(
    document: TextDocumentWithEntities,
) -> ExampleDocumentWithSimpleSpans:
    result = ExampleDocumentWithSimpleSpans(text=document.text, spans=document.entities)
    for entity in document.entities:
        result.spans.append(Span(start=entity.start, end=entity.end))
    return result


def test_builder_with_document_converters_resolve_document_type_and_converter():
    @dataclass
    class RenamedExampleDocument(TextBasedDocument):
        spans: AnnotationList[LabeledSpan] = annotation_field(target="text")

    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "single_config"))
    builder_cls: Type[PieDatasetBuilder] = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        builder = builder_cls(
            cache_dir=tmp_cache_dir,
            document_converters={
                "tests.data.test_builder.ExampleDocumentWithSimpleSpans": "tests.data.test_builder.convert_example_document_to_example_document_with_simple_spans",
            },
        )
    assert isinstance(builder, PieDatasetBuilder)
    assert builder.document_converters == {
        ExampleDocumentWithSimpleSpans: convert_example_document_to_example_document_with_simple_spans,
    }


class NoDocumentType:
    pass


def test_builder_with_document_converters_resolve_wrong_document_type():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "single_config"))
    builder_cls: Type[PieDatasetBuilder] = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "The key 'tests.data.test_builder.NoDocumentType' for one of the converters can not be resolved to a document type."
            ),
        ):
            builder = builder_cls(
                cache_dir=tmp_cache_dir,
                document_converters={
                    "tests.data.test_builder.NoDocumentType": convert_example_document_to_example_document_with_simple_spans,
                },
            )
