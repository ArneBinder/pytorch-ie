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

## 🔭 Demos

| Task                                                       | Link                                                                                                                                                                  |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Named Entity Recognition (Span-based)                      | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/pie/NER)                               |
| Joint Named Entity Recognition and Relation Classification | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/pie/Joint-NER-and-Relation-Extraction) |

## 🚀️ Quickstart

```console
$ pip install pytorch-ie
```

## ⚡️ Examples: Prediction

**The following examples work out of the box. No further setup like manually downloading a model is needed!**

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

ner_pipeline(document)

for entity in document.entities.predictions:
    print(f"{entity} -> {entity.label}")

# Result:
# IndieBio -> ORG
# Po Bronson -> PER
# SOSV -> ORG
```

<details>
<summary>
To create the same pipeline as above without `AutoPipeline`
</summary>

```python
from pytorch_ie.auto import AutoTaskModule, AutoModel
from pytorch_ie.pipeline import Pipeline

model_name_or_path = "pie/example-ner-spanclf-conll03"
ner_taskmodule = AutoTaskModule.from_pretrained(model_name_or_path)
ner_model = AutoModel.from_pretrained(model_name_or_path)
ner_pipeline = Pipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1, num_workers=0)
```

</details>

<details>
<summary>
Or, without `Auto` classes at all
</summary>

```python
from pytorch_ie.pipeline import Pipeline
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule

model_name_or_path = "pie/example-ner-spanclf-conll03"
ner_taskmodule = TransformerSpanClassificationTaskModule.from_pretrained(model_name_or_path)
ner_model = TransformerSpanClassificationModel.from_pretrained(model_name_or_path)
ner_pipeline = Pipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1, num_workers=0)
```

</details>

<details>
<summary>

### Text-classification-based Relation Extraction

</summary>

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

re_pipeline(document, batch_size=2)

for relation in document.relations.predictions:
    print(f"({relation.head} -> {relation.tail}) -> {relation.label}")

# Result:
# (Po Bronson -> SOSV) -> per:employee_of
# (Po Bronson -> IndieBio) -> per:employee_of
# (SOSV -> Po Bronson) -> org:top_members/employees
# (IndieBio -> Po Bronson) -> org:top_members/employees
```

</details>

## ⚡️ Examples: Training

<details>

<summary>

### Span-classification-based Named Entity Recognition

</summary>

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import datasets
from pytorch_ie.models.transformer_span_classification import TransformerSpanClassificationModel
from pytorch_ie.taskmodules.transformer_span_classification import (
    TransformerSpanClassificationTaskModule,
)

pl.seed_everything(42)

model_output_path = "./model_output/"
model_name = "bert-base-cased"
num_epochs = 10
batch_size = 32

# Get the PIE dataset consisting of PIE Documents that will be used for training (and evaluation).
dataset = datasets.load_dataset(
    path="pie/conll2003",
)
train_docs, val_docs = dataset["train"], dataset["validation"]

print("train docs: ", len(train_docs))
print("val docs: ", len(val_docs))

# Create a PIE taskmodule.
task_module = TransformerSpanClassificationTaskModule(
    tokenizer_name_or_path=model_name,
    max_length=128,
)

# Prepare the taskmodule with the training data. This may collect available labels etc.
# The result of this should affect the state of the taskmodule config which will be
# persisted (and can be loaded) later on.
task_module.prepare(train_docs)

# Persist the taskmodule. This writes the taskmodule config as a json file into the
# model_output_path directory. The config contains all constructor parameters to
# re-create the taskmodule at this state (via AutoTaskmodule.from_pretrained(model_output_path)).
task_module.save_pretrained(model_output_path)

# Use the taskmodule to encode the train and dev sets. This may use the text and
# available annotations of the documents.
train_dataset = task_module.encode(train_docs, encode_target=True, as_dataset=True)
val_dataset = task_module.encode(val_docs, encode_target=True, as_dataset=True)

# Create the dataloaders. Note that the taskmodule provides the collate function!
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=task_module.collate,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=task_module.collate,
)

# Create the PIE model. Note that we use the number of entries in the previously
# collected label_to_id mapping to set the number of classes to predict.
model = TransformerSpanClassificationModel(
    model_name_or_path=model_name,
    num_classes=len(task_module.label_to_id),
    t_total=len(train_dataloader) * num_epochs,
    learning_rate=1e-4,
)

# Optionally, set up a model checkpoint callback. See here for further information:
# https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html
# checkpoint_callback = ModelCheckpoint(
#     monitor="val/f1",
#     dirpath=model_output_path,
#     filename="zs-ner-{epoch:02d}-val_f1-{val/f1:.2f}",
#     save_top_k=1,
#     mode="max",
#     auto_insert_metric_name=False,
#     save_weights_only=True,
# )

# Create the pytorch-lightning trainer. See here for further information:
# https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html
trainer = pl.Trainer(
    fast_dev_run=False,
    max_epochs=num_epochs,
    gpus=0,
    checkpoint_callback=False,
    # callbacks=[checkpoint_callback],
    precision=32,
)
# Start the training.
trainer.fit(model, train_dataloader, val_dataloader)

# Persist the trained model. This will save the model weights and the model config that allows
# to re-create the model at this state (via AutoModel.from_pretrained(model_output_path)).
# model.save_pretrained(model_output_path)
```

</details>

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

<details>
<summary><b>How to create your own Pytorch-IE dataset</b></summary>

PyTorch-IE datasets are built on top of Huggingface datasets. For instance, consider the
[conll2003 from the Huggingface Hub](https://huggingface.co/datasets/conll2003) and especially their respective
[dataset loading script](https://huggingface.co/datasets/conll2003/blob/main/conll2003.py). To create a PyTorch-IE
dataset from that, you have to implement:

1. A Document class. This will be the type of individual dataset examples.

```python
@dataclass
class CoNLL2003Document(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
```

Here we derive from `TextDocument` that has a simple `text` string as base annotation target. The `CoNLL2003Document`
adds one single annotation list called `entities` that consists of `LabeledSpan`s which reference the `text` field of
the document. You can add further annotation types by adding `AnnotationList` fields that may also reference (i.e.
`target`) other annotations as you like. See ['pytorch_ie.annotations`](src/pytorch_ie/annotations.py) for predefined
annotation types.

2. A dataset config. This is similar to
   [creating a Huggingface dataset config](https://huggingface.co/docs/datasets/dataset_script#multiple-configurations).

```python
class CoNLL2003Config(datasets.BuilderConfig):
    """BuilderConfig for CoNLL2003"""

    def __init__(self, **kwargs):
        """BuilderConfig for CoNLL2003.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)
```

3. A dataset builder class. This should inherit from
   [`pytorch_ie.data.builder.GeneratorBasedBuilder`](src/pytorch_ie/data/builder.py) which is a wrapper around the
   [Huggingface dataset builder class](https://huggingface.co/docs/datasets/v2.4.0/en/package_reference/builder_classes#datasets.GeneratorBasedBuilder)
   with some utility functionality to work with PyTorch-IE `Documents`. The key elements to implement are: `DOCUMENT_TYPE`,
   `BASE_DATASET_PATH`, and `_generate_document`.

```python
class Conll2003(pytorch_ie.data.builder.GeneratorBasedBuilder):
    # Specify the document type. This will be the class of individual dataset examples.
    DOCUMENT_TYPE = CoNLL2003Document

    # The Huggingface identifier that points to the base dataset. This may be any string that works
    # as path with Huggingface `datasets.load_dataset`.
    BASE_DATASET_PATH = "conll2003"

    # The builder configs, see https://huggingface.co/docs/datasets/dataset_script for further information.
    BUILDER_CONFIGS = [
        CoNLL2003Config(
            name="conll2003", version=datasets.Version("1.0.0"), description="CoNLL2003 dataset"
        ),
    ]

    # [Optional] Define additional keyword arguments which will be passed to `_generate_document` below.
    def _generate_document_kwargs(self, dataset):
        return {"int_to_str": dataset.features["ner_tags"].feature.int2str}

    # Define how a Pytorch-IE Document will be created from a Huggingface dataset example.
    def _generate_document(self, example, int_to_str):
        doc_id = example["id"]
        tokens = example["tokens"]
        ner_tags = [int_to_str(tag) for tag in example["ner_tags"]]

        text, ner_spans = tokens_and_tags_to_text_and_labeled_spans(tokens=tokens, tags=ner_tags)

        document = CoNLL2003Document(text=text, id=doc_id)

        for span in sorted(ner_spans, key=lambda span: span.start):
            document.entities.append(span)

        return document
```

The full script can be found here: [datasets/conll2003/conll2003.py](datasets/conll2003/conll2003.py). Note, that to
load the dataset with `datasets.load_dataset`, the script has to be located in a directory with the same name (as it
is the case for standard Huggingface dataset loading scripts).

</details>

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
