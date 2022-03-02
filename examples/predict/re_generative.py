from pytorch_ie import Document, Pipeline
from pytorch_ie.data import BinaryRelation
from pytorch_ie.models import TransformerSeq2SeqModel
from pytorch_ie.taskmodules import TransformerSeq2SeqTaskModule


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

    document = Document(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    pipeline(document, predict_field="relations")

    relation: BinaryRelation
    for relation in document.predictions["relations"]:
        head, tail = relation.head, relation.tail
        head_text = document.text[head.start : head.end]
        tail_text = document.text[tail.start : tail.end]
        label = relation.label
        print(f"({head_text} -> {tail_text}) -> {label}")


if __name__ == "__main__":
    main()
