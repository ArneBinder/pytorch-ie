import os
import tempfile

import pytest
from absl.testing import parameterized
from datasets.builder import BuilderConfig, DatasetBuilder
from datasets.download.download_manager import DownloadMode
from datasets.load import dataset_module_factory, import_main_class, load_dataset
from datasets.utils.file_utils import DownloadConfig

from tests import DATASET_BUILDERS_ROOT
from tests.data.dataset_tester import DatasetTester


def test_datasets_dir_and_script_names():
    for dataset_dir in DATASET_BUILDERS_ROOT.iterdir():
        name = dataset_dir.name
        if (
            not name.startswith("__") and len(os.listdir(dataset_dir)) > 0
        ):  # ignore __pycache__ and empty dirs
            # check that the script name is the same as the dir name
            assert os.path.exists(
                os.path.join(dataset_dir, name + ".py")
            ), f"Bad structure for dataset '{name}'. Please check that the directory name is a valid dataset and that the same the same as the dataset script name."

            # if name in _PACKAGED_DATASETS_MODULES:
            #     continue
            # else:
            #     # check that the script name is the same as the dir name
            #     assert os.path.exists(
            #         os.path.join(dataset_dir, name + ".py")
            #     ), f"Bad structure for dataset '{name}'. Please check that the directory name is a valid dataset and that the same the same as the dataset script name."


def get_local_dataset_names():
    dataset_script_files = list(DATASET_BUILDERS_ROOT.absolute().glob("**/*.py"))
    datasets = [
        dataset_script_file.parent.name
        for dataset_script_file in dataset_script_files
        if dataset_script_file.name != "__init__.py"
    ]
    return [{"testcase_name": x, "dataset_name": x} for x in datasets]


@parameterized.named_parameters(get_local_dataset_names())
# @for_all_test_methods(skip_if_dataset_requires_faiss, skip_if_not_compatible_with_windows)
class LocalDatasetTest(parameterized.TestCase):
    dataset_name = None

    def setUp(self):
        self.dataset_tester = DatasetTester(self)

    def test_load_dataset(self, dataset_name):
        configs = self.dataset_tester.load_all_configs(dataset_name, is_local=True)[:1]
        self.dataset_tester.check_load_dataset(
            dataset_name, configs, is_local=True, use_local_dummy_data=True
        )

    def test_builder_class(self, dataset_name):
        builder_cls = self.dataset_tester.load_builder_class(dataset_name, is_local=True)
        name = builder_cls.BUILDER_CONFIGS[0].name if builder_cls.BUILDER_CONFIGS else None
        with tempfile.TemporaryDirectory() as tmp_cache_dir:
            builder = builder_cls(config_name=name, cache_dir=tmp_cache_dir)
            self.assertIsInstance(builder, DatasetBuilder)

    def test_builder_configs(self, dataset_name):
        builder_configs = self.dataset_tester.load_all_configs(dataset_name, is_local=True)
        self.assertTrue(len(builder_configs) > 0)

        if builder_configs[0] is not None:
            all(self.assertIsInstance(config, BuilderConfig) for config in builder_configs)

    @pytest.mark.slow
    def test_load_dataset_all_configs(self, dataset_name):
        configs = self.dataset_tester.load_all_configs(dataset_name, is_local=True)
        self.dataset_tester.check_load_dataset(
            dataset_name, configs, is_local=True, use_local_dummy_data=True
        )

    @pytest.mark.slow
    def test_load_real_dataset(self, dataset_name):
        path = str(DATASET_BUILDERS_ROOT / dataset_name)
        dataset_module = dataset_module_factory(
            path, download_config=DownloadConfig(local_files_only=True)
        )
        builder_cls = import_main_class(dataset_module.module_path)
        name = builder_cls.BUILDER_CONFIGS[0].name if builder_cls.BUILDER_CONFIGS else None
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            dataset = load_dataset(
                path,
                name=name,
                cache_dir=temp_cache_dir,
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )
            for split in dataset.keys():
                self.assertTrue(len(dataset[split]) > 0)
            del dataset

    @pytest.mark.slow
    def test_load_real_dataset_all_configs(self, dataset_name):
        path = str(DATASET_BUILDERS_ROOT / dataset_name)
        dataset_module = dataset_module_factory(
            path, download_config=DownloadConfig(local_files_only=True)
        )
        builder_cls = import_main_class(dataset_module.module_path)
        config_names = (
            [config.name for config in builder_cls.BUILDER_CONFIGS]
            if len(builder_cls.BUILDER_CONFIGS) > 0
            else [None]
        )
        for name in config_names:
            with tempfile.TemporaryDirectory() as temp_cache_dir:
                dataset = load_dataset(
                    path,
                    name=name,
                    cache_dir=temp_cache_dir,
                    download_mode=DownloadMode.FORCE_REDOWNLOAD,
                )
                for split in dataset.keys():
                    self.assertTrue(len(dataset[split]) > 0)
                del dataset
