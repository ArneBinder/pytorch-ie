import abc
import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Union, get_type_hints

import datasets

# import pytorch_lightning as pl
from datasets import load_dataset

from pytorch_ie.data.annotations import AnnotationList, BinaryRelation, LabeledSpan, Span
from pytorch_ie.data.document import TextDocument, annotation_field

# from pytorch_ie.models import TransformerTokenClassificationModel
# from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule
from tests import FIXTURES_ROOT

# from datasets.load import extend_dataset_builder_for_streaming, load_dataset_builder
# from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES
# from torch.utils.data import DataLoader


@dataclass
class MyDocument(TextDocument):
    sentences: AnnotationList[Span] = annotation_field(target="text")
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
    # TODO: how to handle this case?
    # topic: Annotation[Label] = ???


def test_create_static_document():
    # document = MyDocument(text="Entity A works at B.")
    document = MyDocument(text="Entity A works at B.", id="ABC123")

    sentence1 = Span(start=0, end=20)

    entity1 = LabeledSpan(start=0, end=8, label="PER")
    entity2 = LabeledSpan(start=18, end=19, label="ORG")

    relation1 = BinaryRelation(head=entity1, tail=entity2, label="per:employee_of")

    # this should fail because head and tail entity annotations are not in the document
    # document.relations.append(relation1)

    # this should work
    document.entities.append(entity1)
    document.entities.append(entity2)
    document.relations.append(relation1)

    print(document.asdict())

    print(MyDocument.fromdict(document.asdict()))

    assert MyDocument.fromdict(document.asdict()) == document
    assert MyDocument.fromdict(document.asdict()).asdict() == document.asdict()

    # assert document.asdict() == {
    #     "text": "Entity A works at B.",
    #     "id": "ABC123",
    #     "sentences": [
    #         {"start": 0, "end": 20, "id": 1}
    #     ],
    #     "entities": [
    #         {"start": 0, "end": 8, "label": "PER", "id": 2},
    #         {"start": 18, "end": 19, "label": "ORG", "id": 3},
    #     ],
    #     "relations": [
    #         {"head": 2, "tail": 3, "label": "per:employee_of", "id": 4}
    #     ],
    # }

    # I'm not sure what to do in this case (because entity1 is referenced by relation1)
    # document.entities.remove(entity1)

    # this should work
    # document.relations.remove(relation1)
    # document.entities.remove(entity1)


def test_load_with_datasets():
    dataset_dir = FIXTURES_ROOT / "datasets" / "json_2"

    dataset = load_dataset(
        # path="json",
        path=str(FIXTURES_ROOT / "datasets" / "json_2" / "json2.py"),
        field="data",
        data_files={
            "train": str(dataset_dir / "train.json"),
            "validation": str(dataset_dir / "val.json"),
            "test": str(dataset_dir / "test.json"),
        },
    )

    print(dataset)

    def convert_to_doc_dict(example):
        doc = MyDocument(text=example["text"], id=example["id"])

        # doc.metadata = dict(example["metadata"])

        sentences = [Span.fromdict(dct) for dct in example["sentences"]]
        entities = [LabeledSpan.fromdict(dct) for dct in example["entities"]]
        relations = [
            BinaryRelation(
                head=entities[rel["head"]], tail=entities[rel["tail"]], label=rel["label"]
            )
            for rel in example["relations"]
        ]

        for sentence in sentences:
            doc.sentences.append(sentence)

        for entity in entities:
            doc.entities.append(entity)

        for relation in relations:
            doc.relations.append(relation)

        # this should be done transparently (but conceptionally it works)
        return doc.asdict()

    train_dataset = dataset["train"].map(convert_to_doc_dict)

    # train_dataset.set_transform(MyDocument.fromdict)

    train_dataset.set_format("document", document_type=MyDocument)

    print(train_dataset[0])

    print(train_dataset[0:2])

    print(train_dataset[0, 1, 2])

    def some_func(document):
        print(document.asdict())
        return document

    train_dataset2 = train_dataset.map(some_func, as_documents=True)
    train_dataset2.set_format("document", document_type=MyDocument)

    print(train_dataset2)

    def some_func_batched(documents):
        for doc in documents:
            doc.relations.clear()
        return documents

    train_dataset3 = train_dataset2.map(
        some_func_batched, as_documents=True, batch_size=2, batched=True
    )
    train_dataset3.set_format("document", document_type=MyDocument)

    print(train_dataset3)


def test_load_with_datasets_conll2003():
    dataset = load_dataset(
        path=str(FIXTURES_ROOT / "datasets" / "conll2003.py"),
    )

    print(dataset)


# def test_load_dataset_and_train():
#     pass
# pl.seed_everything(42)

# model_name = "bert-base-cased"
# num_epochs = 1
# batch_size = 32

# task_module = TransformerTokenClassificationTaskModule(
#     tokenizer_name_or_path=model_name,
#     max_length=128,
# )

# model = TransformerTokenClassificationModel(
#     model_name_or_path=model_name,
#     num_classes=len(task_module.label_to_id),
#     learning_rate=1e-4,
# )

# document_dataset is cached in document format (serialized as a dict of primitive types)
# TODO: create our own load_dataset function (same interface as HF load_dataset, and mostly same logic)
# - ideally we'd have our own custom loading scripts that use the original CoNLL03 dataset loading script
#   provided by HF datasets and only extend it with functionality to convert it to a document
# - the challenge is to find a generic way that allows us to serialize and deserialize a Document.
#   the main problem is that annotations target specific fields, e.g. entities target text, relations target entities,
#   and this must be encoded during serialization, or stored alongside the serialized data
# - this leads to another challenge: how to handle the schema definition when using single documents and a pipeline
# document_dataset = load_dataset(path="conll2003", name="...")

# pie.schema.Document(
#     {
#         "id": pie.schema.Value("string"),
#         "text": pie.schema.Value("string"),
#         "sentences": pie.schema.Sequence(pie.schema.Span(target="text")),
#         "entities": pie.schema.Sequence(pie.schema.LabeledSpan(target="text", class_labels=...)),
#         "relations": pie.schema.Sequence(pie.schema.BinaryRelation(target="entities", class_labels=...)),
#     }
# )

# document_dataset is a Dataset or DatasetDict, or an IterableDataset or IterableDatasetDict
# document_1 = document_dataset["train"][0]

# this may be an option but I don't think it's worth the effort
# train_dataset = document_dataset["train"]
# train_dataset.map(lambda documents: task_module.encode(documents, encode_target=True), batched=True)

# train_documents = document_dataset["train"]
# # TODO: TaskModule.encode should be able to work with iterables as well
# train_dataset = task_module.encode(train_documents, encode_target=True)

# train_dataloader = DataLoader(
#     train_dataset,  # type: ignore
#     batch_size=batch_size,
#     shuffle=True,
#     collate_fn=task_module.collate,
# )

# trainer = pl.Trainer(
#     fast_dev_run=False,
#     max_epochs=num_epochs,
#     gpus=0,
#     checkpoint_callback=False,
#     precision=32,
# )
# trainer.fit(model, train_dataloader)
