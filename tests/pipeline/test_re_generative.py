from dataclasses import dataclass

import pytest

from pytorch_ie import AnnotationList, Pipeline, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextDocument
from pytorch_ie.models import TransformerSeq2SeqModel
from pytorch_ie.taskmodules import TransformerSeq2SeqTaskModule


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@pytest.mark.slow
def test_re_generative():
    model_name_or_path = "Babelscape/rebel-large"

    taskmodule = TransformerSeq2SeqTaskModule(
        tokenizer_name_or_path=model_name_or_path,
        max_input_length=128,
        max_target_length=128,
    )

    model = TransformerSeq2SeqModel(
        model_name_or_path=model_name_or_path,
    )

    pipeline = Pipeline(model=model, taskmodule=taskmodule, device=-1)

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    pipeline(document, predict_field="relations", batch_size=2)

    relations = document.relations.predictions
    assert len(relations) == 2
    sorted_relations = sorted(relations, key=lambda rel: (rel.head.start + rel.tail.start) / 2)

    relation1 = sorted_relations[0]
    assert relation1.label == "subsidiary"
    assert relation1.score == pytest.approx(1.0)
    assert (relation1.head.start, relation1.head.end) == (96, 100)
    assert (relation1.tail.start, relation1.tail.end) == (126, 134)

    relation2 = sorted_relations[1]
    assert relation2.label == "parent organization"
    assert relation2.score == pytest.approx(1.0)
    assert (relation2.head.start, relation2.head.end) == (126, 134)
    assert (relation2.tail.start, relation2.tail.end) == (96, 100)
