from pytorch_ie import Pipeline
from pytorch_ie.data import Document, LabeledSpan
from pytorch_ie.models import TransformerTextClassificationModel
from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule


def main():
    model_name_or_path = "pie/example-re-textclf-tacred"
    re_taskmodule = TransformerRETextClassificationTaskModule.from_pretrained(model_name_or_path)
    re_model = TransformerTextClassificationModel.from_pretrained(model_name_or_path)

    re_pipeline = Pipeline(model=re_model, taskmodule=re_taskmodule, device=-1)

    document = Document(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    for start, end, label in [(65, 75, "PER"), (96, 100, "ORG"), (126, 134, "ORG")]:
        document.add_annotation("entities", LabeledSpan(start, end, label))

    re_pipeline(document, predict_field="relations", batch_size=2)

    for relation in document.predictions.binary_relations["relations"]:
        head, tail = relation.head, relation.tail
        head_text = document.text[head.start : head.end]
        tail_text = document.text[tail.start : tail.end]
        label = relation.label
        print(f"({head_text} -> {tail_text}) -> {label}")


if __name__ == "__main__":
    main()
