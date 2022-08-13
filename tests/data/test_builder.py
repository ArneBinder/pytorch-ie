import tempfile

import pytest
from datasets.load import dataset_module_factory, import_main_class

from datasets import DatasetBuilder
from tests import FIXTURES_ROOT

DATASETS_ROOT = FIXTURES_ROOT / "builder" / "datasets"


def test_builder_class():
    dataset_module = dataset_module_factory(str(DATASETS_ROOT / "single_config"))
    builder_cls = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        builder = builder_cls(cache_dir=tmp_cache_dir)
        assert isinstance(builder, DatasetBuilder)


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


def test_builder_class_name_mapping_disable_but_defaults():
    dataset_module = dataset_module_factory(
        str(DATASETS_ROOT / "name_mapping_disable_but_defaults")
    )
    builder_cls = import_main_class(dataset_module.module_path)
    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        builder = builder_cls(config_name="es", cache_dir=tmp_cache_dir)
        assert builder.info.config_name == "es"
        assert builder.base_builder.info.config_name == "nl"

        builder = builder_cls(config_name="nl", cache_dir=tmp_cache_dir)
        assert builder.info.config_name == "nl"
        assert builder.base_builder.info.config_name == "nl"
