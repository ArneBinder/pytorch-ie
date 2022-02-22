import os

from pytorch_ie.core.hf_hub_mixin import AutoTaskmodule
from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule
from tests import FIXTURES_ROOT


def test_load():
    path = os.path.join(FIXTURES_ROOT, "taskmodules/transformer_re_text_classification")
    taskmodule = AutoTaskmodule.from_pretrained(pretrained_model_name_or_path=path)
    assert isinstance(taskmodule, TransformerRETextClassificationTaskModule)
