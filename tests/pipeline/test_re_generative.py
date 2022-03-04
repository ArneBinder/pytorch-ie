from typing import List

import pytest

from pytorch_ie import Document, Pipeline
from pytorch_ie.data import BinaryRelation
from pytorch_ie.models import TransformerSeq2SeqModel
from pytorch_ie.taskmodules import TransformerSeq2SeqTaskModule


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

    document = Document(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    pipeline(document, predict_field="relations", batch_size=2)
    relations: List[BinaryRelation] = document.predictions.binary_relations["relations"]
    assert len(relations) == 2
    sorted_relations = sorted(relations, key=lambda rel: (rel.head.start + rel.tail.start) / 2)

    rel0 = sorted_relations[0]
    assert rel0.label_single == "subsidiary"
    assert rel0.score == 1.0
    assert (rel0.head.start, rel0.head.end) == (96, 100)
    assert (rel0.tail.start, rel0.tail.end) == (126, 134)

    rel1 = sorted_relations[1]
    assert rel1.label_single == "parent organization"
    assert rel1.score == 1.0
    assert (rel1.head.start, rel1.head.end) == (126, 134)
    assert (rel1.tail.start, rel1.tail.end) == (96, 100)
