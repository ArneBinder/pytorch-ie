import pytest
from pytorch_ie.auto import AutoTaskModule, AutoModel
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule
from pytorch_ie.models import TransformerSpanClassificationModel


@pytest.mark.slow
def test_auto_taskmodule():
    taskmodule = AutoTaskModule.from_pretrained("pie/example-ner-spanclf-conll03")
    assert isinstance(taskmodule, TransformerSpanClassificationTaskModule)
    assert taskmodule.label_to_id == {"O": 0, "MISC": 1, "ORG": 2, "PER": 3, "LOC": 4}


@pytest.mark.slow
def test_auto_model():
    model = AutoModel.from_pretrained("pie/example-ner-spanclf-conll03")
    assert isinstance(model, TransformerSpanClassificationModel)


@pytest.mark.slow
def test_auto_pipeline():
    assert 1 == 0
