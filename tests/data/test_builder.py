import re
import tempfile

import pytest
from datasets import DatasetBuilder, Version
from datasets.load import dataset_module_factory, import_main_class

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
