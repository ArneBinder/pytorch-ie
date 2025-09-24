from dataclasses import dataclass

import pytest
from pie_core import AnnotationLayer, AutoAnnotationPipeline, Model, TaskModule, annotation_field

from pytorch_ie import PyTorchIEPipeline
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.auto import AutoModel, AutoTaskModule
from pytorch_ie.documents import TextDocument
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule


@TaskModule.register()
class MyTransformerSpanClassificationTaskModule(TransformerSpanClassificationTaskModule):
    pass


@Model.register()
class MyTransformerSpanClassificationModel(TransformerSpanClassificationModel):
    pass


@pytest.mark.slow
def test_auto_taskmodule():
    taskmodule = AutoTaskModule.from_pretrained("pie/example-ner-spanclf-conll03")
    assert isinstance(taskmodule, TransformerSpanClassificationTaskModule)
    assert taskmodule.label_to_id == {"O": 0, "MISC": 1, "ORG": 2, "PER": 3, "LOC": 4}
    assert taskmodule.is_prepared


@pytest.mark.slow
def test_auto_taskmodule_full_cycle(tmp_path):

    taskmodule = MyTransformerSpanClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased"
    )
    taskmodule.prepare([])
    taskmodule.save_pretrained(save_directory=str(tmp_path))

    taskmodule_loaded = AutoTaskModule.from_pretrained(str(tmp_path))
    assert isinstance(taskmodule_loaded, MyTransformerSpanClassificationTaskModule)


@pytest.mark.slow
def test_auto_taskmodule_full_cycle_with_name(tmp_path):
    @TaskModule.register(name="mytaskmodule")
    class MyTransformerSpanClassificationTaskModule(TransformerSpanClassificationTaskModule):
        pass

    taskmodule = MyTransformerSpanClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased"
    )
    taskmodule.prepare([])
    assert taskmodule.config["taskmodule_type"] == "mytaskmodule"

    taskmodule.save_pretrained(save_directory=str(tmp_path))

    taskmodule_loaded = AutoTaskModule.from_pretrained(str(tmp_path))
    assert isinstance(taskmodule_loaded, MyTransformerSpanClassificationTaskModule)


def test_auto_taskmodule_from_config():

    config = {
        "taskmodule_type": "MyTransformerSpanClassificationTaskModule",
        "tokenizer_name_or_path": "bert-base-uncased",
        "label_to_id": {"O": 0, "MISC": 1, "ORG": 2, "PER": 3, "LOC": 4},
    }
    taskmodule = AutoTaskModule.from_config(config)
    assert isinstance(taskmodule, MyTransformerSpanClassificationTaskModule)
    assert taskmodule.is_prepared


@pytest.mark.slow
def test_auto_model():
    model = AutoModel.from_pretrained("pie/example-ner-spanclf-conll03")
    assert isinstance(model, TransformerSpanClassificationModel)


def test_auto_model_from_config():
    config = {
        "model_type": "MyTransformerSpanClassificationModel",
        "model_name_or_path": "prajjwal1/bert-tiny",
        "num_classes": 5,
    }
    model = AutoModel.from_config(config)
    assert isinstance(model, MyTransformerSpanClassificationModel)


def test_auto_model_full_cycle(tmp_path):

    model = MyTransformerSpanClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny", num_classes=5
    )
    assert model._config() == {
        "model_type": "MyTransformerSpanClassificationModel",
        "model_name_or_path": "prajjwal1/bert-tiny",
        "num_classes": 5,
        "learning_rate": 1e-05,
        "task_learning_rate": 0.0001,
        "warmup_proportion": 0.1,
        "ignore_index": 0,
        "max_span_length": 8,
        "span_length_embedding_dim": 150,
    }

    model.save_pretrained(save_directory=str(tmp_path))

    model_loaded = AutoModel.from_pretrained(str(tmp_path))
    assert isinstance(model_loaded, MyTransformerSpanClassificationModel)


def test_auto_model_full_cycle_with_name(tmp_path):
    @Model.register("other_model")
    class MyOtherTransformerSpanClassificationModel(TransformerSpanClassificationModel):
        pass

    model = MyOtherTransformerSpanClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny", num_classes=5
    )
    assert model._config() == {
        "model_type": "other_model",
        "model_name_or_path": "prajjwal1/bert-tiny",
        "num_classes": 5,
        "learning_rate": 1e-05,
        "task_learning_rate": 0.0001,
        "warmup_proportion": 0.1,
        "ignore_index": 0,
        "max_span_length": 8,
        "span_length_embedding_dim": 150,
    }

    model.save_pretrained(save_directory=str(tmp_path))

    model_loaded = AutoModel.from_pretrained(str(tmp_path))
    assert isinstance(model_loaded, MyOtherTransformerSpanClassificationModel)


@pytest.mark.slow
def test_auto_pipeline():
    @dataclass
    class ExampleDocument(TextDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    pipeline = PyTorchIEPipeline.from_pretrained("pie/example-ner-spanclf-conll03")
    # TODO: this requires fix in pie_core.Auto.from_config to work when pipeline_type is passed, but not in the config
    # pipeline = AutoAnnotationPipeline.from_pretrained("pie/example-ner-spanclf-conll03", pipeline_type="pytorch-ie")

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    pipeline(document, num_workers=0)

    entities = document.entities.predictions
    assert len(entities) == 3


@pytest.mark.slow
def test_auto_pipeline_full_cycle(tmp_path):
    @dataclass
    class ExampleDocument(TextDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    pipeline = PyTorchIEPipeline.from_pretrained("pie/example-ner-spanclf-conll03")

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    pipeline(document, num_workers=0)

    entities = document.entities.predictions
    assert len(entities) == 3

    pipeline.save_pretrained(save_directory=str(tmp_path))

    pipeline_loaded = PyTorchIEPipeline.from_pretrained(str(tmp_path))
    assert isinstance(pipeline_loaded, PyTorchIEPipeline)

    document2 = ExampleDocument(text=document.text)
    pipeline_loaded(document2, num_workers=0)
    entities2 = document2.entities.predictions
    assert len(entities2) == 3
    assert entities2.resolve() == entities.resolve()
