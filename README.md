# PyTorchIE: State-of-the-art Information Extraction in PyTorch

## 🚀&nbsp;&nbsp;Quickstart

```sh
pip install git+ssh://git@github.com/ChristophAlt/pytorch-ie.git
```

## Example (Span-classification-based Named Entity Recognition)

```python
from pytorch_ie.taskmodules.transformer_span_classification import TransformerSpanClassificationTaskModule
from pytorch_ie.models.transformer_span_classification import TransformerSpanClassificationModel
from pytorch_ie.pipeline import Pipeline
from pytorch_ie.data.document import Document

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
