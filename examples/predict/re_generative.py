from dataclasses import dataclass

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.models import TransformerSeq2SeqModel
from pytorch_ie.pipeline import PyTorchIEPipeline
from pytorch_ie.taskmodules import TransformerSeq2SeqTaskModule


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


def main():
    model_name_or_path = "Babelscape/rebel-large"

    taskmodule = TransformerSeq2SeqTaskModule(
        tokenizer_name_or_path=model_name_or_path,
        entity_annotation="entities",
        relation_annotation="relations",
        max_input_length=128,
        max_target_length=128,
    )

    model = TransformerSeq2SeqModel(
        model_name_or_path=model_name_or_path,
    )

    pipeline = PyTorchIEPipeline(model=model, taskmodule=taskmodule, device=-1)

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    pipeline(document)

    for relation in document.relations.predictions:
        print(f"({relation.head} -> {relation.tail}) -> {relation.label}")


if __name__ == "__main__":
    main()
