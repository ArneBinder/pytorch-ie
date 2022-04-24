from dataclasses import dataclass

from pytorch_ie import (
    AnnotationList,
    BinaryRelation,
    LabeledSpan,
    Pipeline,
    TextDocument,
    annotation_field,
)
from pytorch_ie.models import TransformerTextClassificationModel
from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


def main():
    model_name_or_path = "pie/example-re-textclf-tacred"
    re_taskmodule = TransformerRETextClassificationTaskModule.from_pretrained(model_name_or_path)
    re_model = TransformerTextClassificationModel.from_pretrained(model_name_or_path)

    re_pipeline = Pipeline(model=re_model, taskmodule=re_taskmodule, device=-1)

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    for start, end, label in [(65, 75, "PER"), (96, 100, "ORG"), (126, 134, "ORG")]:
        document.entities.append(LabeledSpan(start=start, end=end, label=label))

    re_pipeline(document, predict_field="relations", batch_size=2)

    for relation in document.relations.predictions:
        print(f"({relation.head} -> {relation.tail}) -> {relation.label}")


if __name__ == "__main__":
    main()
