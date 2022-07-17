# PyTorch-IE: State-of-the-art Information Extraction in PyTorch

[![PyPI](https://img.shields.io/pypi/v/pytorch-ie.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/pytorch-ie.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/pytorch-ie)][pypi status]
[![License](https://img.shields.io/pypi/l/pytorch-ie)][license]

[![Read the documentation at https://pytorch-ie.readthedocs.io/](https://img.shields.io/readthedocs/pytorch-ie/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/christophalt/pytorch-ie/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/christophalt/pytorch-ie/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/pytorch-ie/
[read the docs]: https://pytorch-ie.readthedocs.io/
[tests]: https://github.com/christophalt/pytorch-ie/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/christophalt/pytorch-ie
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## 🤯 What's this about?

This is an experimental framework that aims to combine the lessons learned from five years of information extraction research.

-   **Focus on the core task:** The main goal is to develop information extraction methods not dataset loading and evaluation logic. We use external well-maintained libraries for non-core functionality. PyTorch-Lightning for training and logging, Huggingface datasets for dataset reading, and Huggingface evaluate for evaluation (coming soon).
-   **Sharing is caring:** Being able to quickly and easily share models is key to promote your work and facilitate further research. All models developed in PyTorch-IE can be easily shared via the Huggingface model hub. This further allows to quickly build demos based on Huggingface spaces, gradio or streamlit.
-   **Unified document format:** A unified document format allows for quick experimentation on any dataset or task.
-   **Beyond sentence level:** Most information extraction frameworks assume text inputs at a sentence granularity. We do not make any assumption on the granularity but generally aim for document-level information extraction.
-   **Beyond unstructured text:** Unstructured text is only one possible area for information extraction. We developed the framework to also support information extraction from semi-structured text (e.g. HTML), two-dimensional text (e.g. OCR'd images), and images.
-   **Character-level annotation and evaluation:** Many information extraction frameworks annotate and evaluate on a token level. We believe that annotation and evaluation should be done on a character level as this also considers the suitability of the tokenizer for the task.
-   **Make no assumptions on the structure of models:** The last years have seen many different and creative approaches to information extraction and a framework that imposes a structure on those will most certainly be to limiting. With PyTorch-iE you have full control over how a document is prepared for a model and how the model is structured. The logic is self-contained and thus can be easily shared and inspected by others. The only assumption we make is that the input is a document and the output are targets (training) or annotations (inference).

## 🚀️ Quickstart

```console
$ pip install pytorch-ie
```

## 🔭 Demos

| Task                                                       | Link (Huggingface Spaces)                                                   |
| ---------------------------------------------------------- | --------------------------------------------------------------------------- |
| Named Entity Recognition (Span-based)                      | [LINK](https://huggingface.co/spaces/pie/NER)                               |
| Joint Named Entity Recognition and Relation Classification | [LINK](https://huggingface.co/spaces/pie/Joint-NER-and-Relation-Extraction) |

## 📚 Datasets

We parse all datasets into a common format that can be loaded directly from the model hub via Huggingface datasets. The documents are cached in an arrow table and serialized / deserialized on the fly. Any changes or preprocessing applied to the documents will be cached as well.

```python
import datasets

dataset = datasets.load_dataset("pie/conll2003")

print(dataset["train"][0])
# >>> CoNLL2003Document(text='EU rejects German call to boycott British lamb .', id='0', metadata={})

dataset["train"][0].entities
# >>> AnnotationList([LabeledSpan(start=0, end=2, label='ORG', score=1.0), LabeledSpan(start=11, end=17, label='MISC', score=1.0), LabeledSpan(start=34, end=41, label='MISC', score=1.0)])

entity = dataset["train"][0].entities[1]

print(f"[{entity.start}, {entity.end}] {entity}")
# >>> [11, 17] German
```

## ⚡️ Example

**Note:** Setting `num_workers=0` in the pipeline is only necessary when running an example in an
interactive python session. The reason is that multiprocessing doesn't play well with the interactive python
interpreter, see [here](https://docs.python.org/3/library/multiprocessing.html#using-a-pool-of-workers)
for details.

### Span-classification-based Named Entity Recognition

```python
from dataclasses import dataclass

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.auto import AutoPipeline
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument

@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")

document = ExampleDocument(
    "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
)

# see below for the long version
ner_pipeline = AutoPipeline.from_pretrained("pie/example-ner-spanclf-conll03", device=-1, num_workers=0)

ner_pipeline(document, predict_field="entities")

for entity in document.entities.predictions:
    print(f"{entity} -> {entity.label}")

# Result:
# IndieBio -> ORG
# Po Bronson -> PER
# SOSV -> ORG
```

To create the same pipeline as above without `AutoPipeline`:

```python
from pytorch_ie.auto import AutoTaskModule, AutoModel
from pytorch_ie.pipeline import Pipeline

model_name_or_path = "pie/example-ner-spanclf-conll03"
ner_taskmodule = AutoTaskModule.from_pretrained(model_name_or_path)
ner_model = AutoModel.from_pretrained(model_name_or_path)
ner_pipeline = Pipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1, num_workers=0)
```

Or, without `Auto` classes at all:

```python
from pytorch_ie.pipeline import Pipeline
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule

model_name_or_path = "pie/example-ner-spanclf-conll03"
ner_taskmodule = TransformerSpanClassificationTaskModule.from_pretrained(model_name_or_path)
ner_model = TransformerSpanClassificationModel.from_pretrained(model_name_or_path)
ner_pipeline = Pipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1, num_workers=0)
```

## ⚡️️️️ More Examples

### Text-classification-based Relation Extraction

```python
from dataclasses import dataclass

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.auto import AutoPipeline
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

document = ExampleDocument(
    "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
)

re_pipeline = AutoPipeline.from_pretrained("pie/example-re-textclf-tacred", device=-1, num_workers=0)

for start, end, label in [(65, 75, "PER"), (96, 100, "ORG"), (126, 134, "ORG")]:
    document.entities.append(LabeledSpan(start=start, end=end, label=label))

re_pipeline(document, predict_field="relations", batch_size=2)

for relation in document.relations.predictions:
    print(f"({relation.head} -> {relation.tail}) -> {relation.label}")

# Result:
# (Po Bronson -> SOSV) -> per:employee_of
# (Po Bronson -> IndieBio) -> per:employee_of
# (SOSV -> Po Bronson) -> org:top_members/employees
# (IndieBio -> Po Bronson) -> org:top_members/employees
```

<!-- github-only -->

✨📚✨ [Read the full documentation](https://pytorch-ie.readthedocs.io/)

## 🔧 Development Setup

## 🏅 Acknowledgements

-   This package is based on the [sourcery-ai/python-best-practices-cookiecutter](https://github.com/sourcery-ai/python-best-practices-cookiecutter) and [cjolowicz/cookiecutter-hypermodern-python](https://github.com/cjolowicz/cookiecutter-hypermodern-python) project templates.

## 📃 Citation

If you find the framework useful please consider citing it:

```bibtex
@misc{alt2022pytorchie,
    author={Christoph Alt, Arne Binder},
    title = {PyTorch-IE: State-of-the-art Information Extraction in PyTorch},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ChristophAlt/pytorch-ie}}
}
```

[license]: https://github.com/christophalt/pytorch-ie/blob/main/LICENSE
