from dataclasses import dataclass

from pytorch_ie import AnnotationList, LabeledSpan, TextDocument, annotation_field
from pytorch_ie.auto import AutoPipeline


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


def main():

    ner_pipeline = AutoPipeline.from_pretrained("pie/example-ner-spanclf-conll03", device=-1)

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    ner_pipeline(document, predict_field="entities")

    for entity in document.entities.predictions:
        print(f"{entity} -> {entity.label}")


if __name__ == "__main__":
    main()
