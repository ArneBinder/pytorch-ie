# PyTorch-IE: State-of-the-art Information Extraction in PyTorch

## 🚀&nbsp;&nbsp;Quickstart

```sh
pip install git+ssh://git@github.com/ChristophAlt/pytorch-ie.git
```

## ⚡&nbsp;&nbsp;Example

#### Span-classification-based Named Entity Recognition

```python
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie import Pipeline, Document

model_name_or_path = "pie/example-ner-spanclf-conll03"
ner_taskmodule = TransformerSpanClassificationTaskModule.from_pretrained(model_name_or_path)
ner_model = TransformerSpanClassificationModel.from_pretrained(model_name_or_path)

ner_pipeline = Pipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1)

document = Document("“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio.")

ner_pipeline(document, predict_field="entities")

for entity in document.predictions("entities"):
    entity_text = document.text[entity.start: entity.end]
    label = entity.label
    print(f"{entity_text} -> {label}")

# Result:
# IndieBio -> ORG
# Po Bronson -> PER
# SOSV -> ORG
```

### More examples

<details>
<summary><b>Text-classification-based Relation Extraction</b></summary>

```python
from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule
from pytorch_ie.models import TransformerTextClassificationModel
from pytorch_ie import Pipeline
from pytorch_ie.data import Document, LabeledSpan

model_name_or_path = "pie/example-re-textclf-tacred"
re_taskmodule = TransformerRETextClassificationTaskModule.from_pretrained(model_name_or_path)
re_model = TransformerTextClassificationModel.from_pretrained(model_name_or_path)

re_pipeline = Pipeline(model=re_model, taskmodule=re_taskmodule, device=-1)

document = Document("“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio.")

for start, end, label in [(65, 75, "PER"), (96, 100, "ORG"), (126, 134, "ORG")]:
    document.add_annotation("entities", LabeledSpan(start, end, label))

re_pipeline(document, predict_field="relations")

for relation in document.predictions("relations"):
    head, tail = relation.head, relation.tail
    head_text = document.text[head.start: head.end]
    tail_text = document.text[tail.start: tail.end]
    label = relation.label
    print(f"({head_text} -> {tail_text}) -> {label}")

# Result:
# (Po Bronson -> SOSV) -> per:employee_of
# (Po Bronson -> IndieBio) -> per:employee_of
# (SOSV -> Po Bronson) -> org:top_members/employees
# (IndieBio -> Po Bronson) -> org:top_members/employees
```

</details>

## Development Setup

```sh
# Install dependencies
poetry install

# Setup pre-commit and pre-push hooks
poetry run pre-commit install -t pre-commit
poetry run pre-commit install -t pre-push
```

## Credits

This package was created with Cookiecutter and the [sourcery-ai/python-best-practices-cookiecutter](https://github.com/sourcery-ai/python-best-practices-cookiecutter) project template.

## BibTeX

If you want to cite the framework feel free to use this:

```bibtex
@article{alt2021pytorchie,
  title={PyTorch-IE},
  author={Christoph Alt, Arne Binder},
  journal={GitHub. Note: https://github.com/ChristophAlt/pytorch-ie},
  year={2021}
}
```
