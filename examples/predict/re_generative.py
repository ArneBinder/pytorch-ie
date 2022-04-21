from dataclasses import dataclass

from pytorch_ie import (
    AnnotationList,
    BinaryRelation,
    LabeledSpan,
    Pipeline,
    TextDocument,
    annotation_field,
)
from pytorch_ie.models import TransformerSeq2SeqModel
from pytorch_ie.taskmodules import TransformerSeq2SeqTaskModule


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


def main():
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

    pipeline(document, predict_field="relations")

    for relation in document.relations.predictions:
        head_text = relation.head.text
        tail_text = relation.tail.text
        label = relation.label
        print(f"({head_text} -> {tail_text}) -> {label}")


if __name__ == "__main__":
    main()
