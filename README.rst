PyTorch-IE: State-of-the-art Information Extraction in PyTorch
==============================================================

.. badges-begin

| |Status| |Python Version| |License| |Read the Docs|
| |Tests| |Codecov| |pre-commit| |Black| |Contributor Covenant|

.. |Status| image:: https://badgen.net/badge/status/alpha/d8624d
   :target: https://badgen.net/badge/status/alpha/d8624d
   :alt: Project Status
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/pytorch-ie
   :target: https://github.com/christophalt/pytorch-ie
   :alt: Python Version
.. |License| image:: https://img.shields.io/github/license/christophalt/pytorch-ie
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/pytorch-ie/latest.svg?label=Read%20the%20Docs
   :target: https://pytorch-ie.readthedocs.io/
   :alt: Read the documentation at https://pytorch-ie.readthedocs.io/
.. |Tests| image:: https://github.com/christophalt/pytorch-ie/workflows/Tests/badge.svg
   :target: https://github.com/christophalt/pytorch-ie/actions?workflow=Tests
   :alt: Tests
.. |Codecov| image:: https://codecov.io/gh/christophalt/pytorch-ie/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/christophalt/pytorch-ie
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black
.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
   :target: https://github.com/christophalt/pytorch-ie/blob/main/CODE_OF_CONDUCT.rst
   :alt: Contributor Covenant

.. badges-end

-----

🚀️ Quickstart
---------------

.. code:: console

    $ pip install pytorch-ie


⚡️ Examples
------------
**Note:** Setting ``num_workers=0`` in the pipeline is only necessary when running an example in an
interactive python session. The reason is that multiprocessing doesn't play well with the interactive python
interpreter, see `here <https://docs.python.org/3/library/multiprocessing.html#using-a-pool-of-workers>`_
for details.

Span-classification-based Named Entity Recognition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

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

To create the same pipeline as above without `AutoPipeline`:

.. code:: python

    from pytorch_ie.auto import AutoTaskModule, AutoModel
    from pytorch_ie.pipeline import Pipeline

    model_name_or_path = "pie/example-ner-spanclf-conll03"
    ner_taskmodule = AutoTaskModule.from_pretrained(model_name_or_path)
    ner_model = AutoModel.from_pretrained(model_name_or_path)
    ner_pipeline = Pipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1, num_workers=0)

Or, without `Auto` classes at all:

.. code:: python

    from pytorch_ie.pipeline import Pipeline
    from pytorch_ie.models import TransformerSpanClassificationModel
    from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule

    model_name_or_path = "pie/example-ner-spanclf-conll03"
    ner_taskmodule = TransformerSpanClassificationTaskModule.from_pretrained(model_name_or_path)
    ner_model = TransformerSpanClassificationModel.from_pretrained(model_name_or_path)
    ner_pipeline = Pipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1, num_workers=0)


Text-classification-based Relation Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

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

..
  github-only

✨📚✨ `Read the full documentation`__

__ https://pytorch-ie.readthedocs.io/

Development Setup
-----------------

🏅 Acknowledgements
---------------------

- This package is based on the `sourcery-ai/python-best-practices-cookiecutter`_ and `cjolowicz/cookiecutter-hypermodern-python`_ project templates.

.. _sourcery-ai/python-best-practices-cookiecutter: https://github.com/sourcery-ai/python-best-practices-cookiecutter
.. _cjolowicz/cookiecutter-hypermodern-python: https://github.com/cjolowicz/cookiecutter-hypermodern-python


📃 Citation
-------------

If you want to cite the framework feel free to use this:

.. code:: bibtex

    @misc{alt2022pytorchie,
    author={Christoph Alt, Arne Binder},
    title = {PyTorch-IE: State-of-the-art Information Extraction in PyTorch},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ChristophAlt/pytorch-ie}}
    }
