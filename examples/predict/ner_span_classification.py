from dataclasses import dataclass

from pie_core import AnnotationLayer, annotation_field
from pie_documents.annotations import LabeledSpan
from pie_documents.documents import TextDocument

from pytorch_ie import PyTorchIEPipeline
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


def main():
    model_name_or_path = "pie/example-ner-spanclf-conll03"
    ner_taskmodule = TransformerSpanClassificationTaskModule.from_pretrained(model_name_or_path)
    ner_model = TransformerSpanClassificationModel.from_pretrained(model_name_or_path)

    ner_pipeline = PyTorchIEPipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1)

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    ner_pipeline(document)

    for entity in document.entities.predictions:
        print(f"{entity} -> {entity.label}")


if __name__ == "__main__":
    main()
