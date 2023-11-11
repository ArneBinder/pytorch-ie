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

**For even faster prototyping with pre-defined, but fully configurable training pipelines and much more useful tooling, have a look into the [PyTorch-IE-Hydra-Template](https://github.com/ChristophAlt/pytorch-ie-hydra-template).**

## 🥧 Concepts & Architecture

PyTorch-IE builds on three core concepts: the **📃 Document**, the **🔤 ⇔ 🔢 Taskmodule**, and the **🧮 Model**. In a
nutshell, the Document says how your data is structured, the Model defines your trainable logic and the Taskmodule
converts from one end to the other. All three concepts are represented as abstract classes that should be used to
derive use-case specific versions. In the following, they are explained in detail.

<details>
<summary>

### 📃 Document

</summary>

The `Document` class is a special `dataclass` that defines the document model. Derivations can contain several
elements:

-   **Data fields** like strings to represent one or multiple texts or arrays for image data. These elements can be
    arbitrary python objects.
-   **Annotation fields** like labeled spans for entities or labeled tuples of spans for relations. These elements have
    to be of a certain container type `AnnotationList` that is dynamically typed with the actual annotation type, e.g.
    `entities: AnnotationList[LabeledSpan]`. Furthermore, annotation elements define one or multiple annotation `targets`.
    An annotation target is either a data element or another annotation container. Internally, targets are used to construct the
    annotation graph, i.e. data elements and annotation containers are the nodes and targets define the edges. The
    annotation graph defines the (de-)serialization order and what is accessible from within an annotation. To
    facilitate the setup of annotation containers, there is the `annotation_field()` method.
-   **Other fields** to save metadata, ids, etc. They are not constrained in any way, but can not be accessed from within
    annotations.

<details>

<summary>

#### Example Document Model

</summary>

```python
from typing import Optional
from pytorch_ie.core import Document, AnnotationLayer, annotation_field
from pytorch_ie.annotations import LabeledSpan, BinaryRelation, Label


class MyDocument(Document):
    # data fields (any field that is targeted by an annotation fields)
    text: str
    # annotation fields
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")
    label: AnnotationLayer[Label] = annotation_field()
    # other fields
    doc_id: Optional[str] = None
```

Note that the `label` is a special annotation field that does not define a target because it belongs to the whole document.
You can also have more complex constructs, like annotation fields that target multiple other fields by using
`annotation_field(targets)` or `annotation_field(named_targets)`. The latter is useful if you want to access the
targets by name from within the annotation, see below for an example.

</details>

#### Annotations

There are several predefined **annotation types** in `pytorch_ie.annotations`, however, feel free to define your own.
Annotations have to be dataclasses that subclass `pytorch_ie.core.Annotation`. They also need to be hashable and
immutable. The following is a simple example:

```python
@dataclass(eq=True, frozen=True)
class SimpleLabeledSpan(Annotation):
    start: int
    end: int
    label: str
```

<details>
<summary>

##### Accessing Target Content

</summary>

We can expand the above example a little to have a nice string representation:

```python
@dataclass(eq=True, frozen=True)
class LabeledSpan(Annotation):
    start: int
    end: int
    label: str

    def __str__(self) -> str:
        if self.targets is None:
            return ""
        return str(self.target[self.start : self.end])
```

The content of `self.target` is lazily assigned as soon as the annotation is added to a document.

Note that this now expects a single `collections.abc.Sequence` as `target`, e.g.:

```python
my_spans: AnnotationLayer[Span] = annotation_field(target="<NAME_OF_THE_SEQUENCE_FIELD>")
```

If we have multiple targets, we need to define target names to access them. For this, we need to set the special
field `TARGET_NAMES`:

```python
@dataclass(eq=True, frozen=True)
class Alignment(Annotation):
    TARGET_NAMES = ("text1", "text2")
    start1: int
    end1: int
    start2: int
    end2: int

    def __str__(self) -> str:
        if self.targets is None:
            return ""
        # we can access the `named_targets` which has the keys defined in `TARGET_NAMES`
        span1 = self.named_targets["text1"][self.start1 : self.end1]
        span2 = self.named_targets["text2"][self.start2 : self.end2]
        return f'span1="{span1}" is aligned with span2="{span2}"'
```

This requires to define the annotation container as follows:

```python
class MyDocumentWithAlignment(Document):
    text_a: str
    text_b: str
    # `named_targets` defines the mapping from `TARGET_NAMES` to data fields
    my_alignments: AnnotationLayer[Alignment] = annotation_field(named_targets={"text1": "text_a", "text2": "text_b"})
```

Note that `text1` and `text2` can also target the same field.

</details>
<details>
<summary>

##### (De-)Serialization of Annotations

</summary>

As usual for dataclasses, annotations can be converted to json like objects with `.asdict()`. However, they can be
also created with `MyAnnotation.fromdict(dct, annotation_store)`. Both methods are required because documents and
their annotations are created on the fly when working with PIE datasets (see below).

</details>
</details>
<details>
<summary>

### 🔤 ⇔ 🔢 Taskmodule

</summary>

The taskmodule is responsible for converting documents to model inputs and back. For that purpose, it requires the
user to implement the following methods:

-   `encode_input`: Taking one document, create one or multiple `TaskEncoding`s. A `TaskEncoding` represents an
    example that will be passed to the model later on. It is a container holding `inputs`, optional `targets`, the
    original `document`, and `metadata`. Note that `encode_input` should not assign a value to `targets`.
-   `encode_target`: This gets a single `TaskEncoding` and should produce a target encoding that will be assigned
    to `targets` later on. As such, it is called only during training / evaluation, but not for inference. Note that,
    this is allowed to return None. In this case, the respective `TaskEncoding` will not be passed to the model at all.
-   `collate`: Taking a batch of `TaskEncoding`s, this should produce a batch input for the model. Note that this has to
    work with available targets (training and evaluation) and without them (inference).
-   `unbatch_output`: This gets a batch output from the model and should rearrange that into a sequence of `TaskOutput`s.
    In that means it can be understood as the opposite to `collate`. The number of `TaskOutput`s should match the
    number of `TaskEncoding`s that got into the batch because we align them later on for easy creation of new annotations.
-   `create_annotations_from_output`: This gets a single `TaskEncoding` with its corresponding `TaskOutput` and
    should yield tuples each consisting of an annotation field name and an annotation. The annotations will be added
    as predictions to the annotation field with the respective name.
-   `prepare` (OPTIONAL): This will get the train dataset, i.e. a Sequence or Iterable of Documents, and can be used
    to calculate additional parameters like the list of all available labels, etc.

You can find some predefined taskmodules for _text-_ and _token classification_, _text classification based relation
extraction_, _joint entity and relation classification_ and other use cases in the package
[`pytorch_ie.taskmodules`](src/pytorch_ie/taskmodules). Especially, have a look at the
[SimpleTransformerTextClassificationTaskModule](src/pytorch_ie/taskmodules/simple_transformer_text_classification.py)
that is well documented and should provide a good starting point to implement your own one.

</details>
<details>
<summary>

### 🧮 Model

</summary>

PyTorch-IE models are meant to do the heavy lifting training and inference. They are
[Pytorch-Lightning modules](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html),
enhanced with some functionality to ease persisting them, see [Reusability and Sharing](#reusability-and-sharing).

You can find some predefined models for transformer based _text-_ and _token classification_, _sequence generation_,
and other use cases in the package [`pytorch_ie.models`](src/pytorch_ie/models).

</details>

### Reusability and Sharing

Taskmodules and Models provide some functionality to ease reusability and reproducibility. Especially, they provide
the methods `save_pretrained()` and `from_pretrained()` that can be used to save their specification, i.e. their
**config**, and available model wights to disc and exactly re-create them again from that data.

<details>
<summary>

#### Huggingface Hub and Extended Configs

</summary>

These methods come along
with integration to the [Huggingface Hub](https://huggingface.co/docs/hub/index). By passing `push_to_hub=True` to
`save_pretrained()`, the taskmodule / model is directly pushed to the Hub and can be loaded again with the respective
identifier (see the [Examples](examples) for how to do so). However, to work properly, each taskmodule / model has to
correctly implement the `_config()` getter method. Per default, it returns all parameters passed to the `__init__`
method if this calls `save_hyperparameters()` which is very recommended. But you may have created some further
parameters that should be persisted, for instance a label-to-id mapping. In this case, `_config()` should be
overwritten to take this into account:

```python
def _config(self) -> Dict[str, Any]:
    # add the label-to-id mapping to the config
    config = super()._config()
    config["label_to_id"] = self.label_to_id
    return config
```

Furthermore, you can use the property `is_from_pretrained` to know if the taskmodule / model is just loaded or created
from scratch. This may be useful, for instance, to avoid downloading a model from Huggingface Transformers when you
in fact want to load your own trained model from disc via `from_pretrained`:

```python
from transformers import AutoConfig, AutoModel

hf_config = AutoConfig.from_pretrained(model_name_or_path)
# If this is already trained, just create an empty transformer model. The weights are loaded afterwards
# via the pytorch_ie.Model.from_pretrained() logic.
if self.is_from_pretrained:
    self.model = AutoModel.from_config(config=hf_config)
# Otherwise, download the whole model from the Huggingface Hub.
else:
    self.model = AutoModel.from_pretrained(model_name_or_path, config=hf_config)
```

</details>

In short, each taskmodule / model implementation should:

-   call `save_hyperparameters()` in `__init__` to save all constructor arguments,
-   pass remaining `__init__` kwargs (keyword arguments) to its super to not break some other helpful functionality
    (e.g. `is_from_pretrained`), and
-   overwrite `_config()` if additional parameters are calculated, e.g. from the dataset.

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
from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import TextDocument


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


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
from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import TextDocument


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


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
# IMPORTANT: This requires pie-datasets >=0.3.0 to be installed! See here for further information:
# https://github.com/ArneBinder/pie-datasets
dataset = datasets.load_dataset(
    path="pie/conll2003",
)
train_docs, val_docs = dataset["train"], dataset["validation"]

print("train docs: ", len(train_docs))
print("val docs: ", len(val_docs))

# Create a PIE taskmodule.
task_module = TransformerSpanClassificationTaskModule(
    tokenizer_name_or_path=model_name,
    entity_annotation="entities",
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
    enable_checkpointing=False,
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

PyTorch-IE works quite well together with Huggingface datasets. Have a look at
[pie-datasets](https://github.com/ArneBinder/pie-datasets) for helpful tooling and a collection of datasets
that are already converted to the PIE format.

<!-- github-only -->

<!-- TODO ✨📚✨ [Read the full documentation](https://pytorch-ie.readthedocs.io/) -->

## 🔧 Project Development

### Setup

This project is build with [Poetry](https://python-poetry.org/). If not yet done, follow the
[Poetry installation guide](https://python-poetry.org/docs/#installation) to install it. Then, install all
dependencies (including for development) for PyTorch-IE by calling the following command from the root of the
repository:

```bash
poetry install --with dev
```

NOTE: If the installation gets stuck, try if disabling experimental parallel installer helps
([source](https://github.com/python-poetry/poetry/issues/3352#issuecomment-732761629)):
`poetry config experimental.new-installer false`

### Testing and code quality checks

We use [Nox](https://nox.thea.codes/en/stable/) to execute any tests and code quality tooling in a reproducible way.

To get a list of available toolchains, call:

```bash
poetry run nox -l
```

To run a specific command from that list, call:

```bash
poetry run nox -s <command>
```

Note: To run the nox commands in the same, reproducible setup that is specified by the lock file, we call them via
`poetry run <nox-command>`.

For instance, to run static type checking with `mypy`, call:

```bash
poetry run nox -s mypy-3.9
```

To run all commands that also run on GitHub CI, call:

```bash
poetry run nox
```

You can also start a shell with Poetry's virtual environment activated by calling:

```bash
poetry shell
```

This allows you to run above commands without the `poetry run` prefix.

### Updating Dependencies

Call this to update individual packages:

```bash
poetry update <package>
```

Then, commit the modified lock file to persist the state.

### Releasing

Since this project is based on the [Cookiecutter template](https://cookiecutter-hypermodern-python.readthedocs.io), we can follow
[their release steps](https://cookiecutter-hypermodern-python.readthedocs.io/en/2022.6.3.post1/guide.html#how-to-make-a-release):

1. Create the release branch:
   `git switch --create release main`
2. Increase the version:
   `poetry version <PATCH|MINOR|MAJOR>`,
   e.g. `poetry version patch` for a patch release
3. Commit the changes:
   `git commit --message="release <NEW VERSION>" pyproject.toml`,
   e.g. `git commit --message="release 0.13.0" pyproject.toml`
4. Push the changes to GitHub:
   `git push origin release`
5. Create a PR for that `release` branch on GitHub.
6. Wait until checks passed successfully.
7. Integrate the PR into the main branch. This triggers the GitHub Action that
   creates all relevant release artefacts and also uploads them to PyPI.
8. Cleanup: Delete the `release` branch.

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
