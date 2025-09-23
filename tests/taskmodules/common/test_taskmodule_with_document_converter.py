from typing import Optional, Type

import pytest
from pie_core import Document
from pie_documents.documents import TextDocumentWithLabeledSpansAndBinaryRelations
from typing_extensions import TypeAlias

from pytorch_ie.taskmodules import RETextClassificationWithIndicesTaskModule
from pytorch_ie.taskmodules.common import TaskModuleWithDocumentConverter
from tests.conftest import TestDocument

DocumentType: TypeAlias = TestDocument
ConvertedDocumentType: TypeAlias = TextDocumentWithLabeledSpansAndBinaryRelations


class MyRETaskModuleWithDocConverter(
    TaskModuleWithDocumentConverter, RETextClassificationWithIndicesTaskModule
):
    @property
    def document_type(self) -> Optional[Type[Document]]:
        return TestDocument

    def _convert_document(self, document: DocumentType) -> ConvertedDocumentType:
        result = document.as_type(
            TextDocumentWithLabeledSpansAndBinaryRelations,
            field_mapping={"entities": "labeled_spans", "relations": "binary_relations"},
        )
        new2old_span = {
            new_s: old_s for old_s, new_s in zip(document.entities, result.labeled_spans)
        }
        result.metadata["new2old_span"] = new2old_span
        return result

    def _integrate_predictions_from_converted_document(
        self, document: DocumentType, converted_document: ConvertedDocumentType
    ) -> None:
        new2old_span = converted_document.metadata["new2old_span"]
        for rel in converted_document.binary_relations.predictions:
            new_rel = rel.copy(head=new2old_span[rel.head], tail=new2old_span[rel.tail])
            document.relations.predictions.append(new_rel)


@pytest.fixture(scope="module")
def taskmodule(documents):
    result = MyRETaskModuleWithDocConverter(tokenizer_name_or_path="bert-base-cased")
    result.prepare(documents)
    return result


def test_taskmodule(taskmodule):
    assert taskmodule is not None
    assert taskmodule.document_type == TestDocument


@pytest.fixture(scope="module")
def task_encodings(taskmodule, documents):
    return taskmodule.encode(documents, encode_target=True)


def test_task_encodings(task_encodings):
    assert len(task_encodings) == 7


def test_decode(taskmodule, task_encodings):
    label_indices = [0, 1, 3, 0, 0, 2, 0]
    probabilities = [0.1738, 0.6643, 0.2101, 0.0801, 0.0319, 0.81, 0.3079]
    task_outputs = [
        {"labels": [taskmodule.id_to_label[label_idx]], "probabilities": [prob]}
        for label_idx, prob in zip(label_indices, probabilities)
    ]
    docs_with_predictions = taskmodule.decode(
        task_encodings=task_encodings, task_outputs=task_outputs
    )
    assert all(isinstance(doc, TestDocument) for doc in docs_with_predictions)

    all_gold_relations = [doc.relations.resolve() for doc in docs_with_predictions]
    assert all_gold_relations == [
        [("per:employee_of", (("PER", "Entity A"), ("ORG", "B")))],
        [
            ("per:employee_of", (("PER", "Entity G"), ("ORG", "H"))),
            ("per:founder", (("PER", "Entity G"), ("ORG", "I"))),
            ("org:founded_by", (("ORG", "I"), ("ORG", "H"))),
        ],
        [
            ("per:employee_of", (("PER", "Entity M"), ("ORG", "N"))),
            ("per:founder", (("PER", "it"), ("ORG", "O"))),
            ("org:founded_by", (("ORG", "O"), ("PER", "it"))),
        ],
    ]

    all_predicted_relations = [
        doc.relations.predictions.resolve() for doc in docs_with_predictions
    ]
    assert all_predicted_relations == [
        [("no_relation", (("PER", "Entity A"), ("ORG", "B")))],
        [
            ("org:founded_by", (("PER", "Entity G"), ("ORG", "H"))),
            ("per:founder", (("PER", "Entity G"), ("ORG", "I"))),
            ("no_relation", (("ORG", "I"), ("ORG", "H"))),
        ],
        [
            ("no_relation", (("PER", "Entity M"), ("ORG", "N"))),
            ("per:employee_of", (("PER", "it"), ("ORG", "O"))),
            ("no_relation", (("ORG", "O"), ("PER", "it"))),
        ],
    ]


class MyRETaskModuleWithDocConverterWithoutDocType(
    TaskModuleWithDocumentConverter, RETextClassificationWithIndicesTaskModule
):
    def _convert_document(self, document: DocumentType) -> ConvertedDocumentType:
        pass

    def _integrate_predictions_from_converted_document(
        self, document: DocumentType, converted_document: ConvertedDocumentType
    ) -> None:
        pass


def test_missing_document_type_overwrite():
    taskmodule = MyRETaskModuleWithDocConverterWithoutDocType(
        tokenizer_name_or_path="bert-base-cased"
    )

    with pytest.raises(NotImplementedError) as e:
        taskmodule.document_type
    assert (
        str(e.value)
        == "please overwrite document_type for MyRETaskModuleWithDocConverterWithoutDocType"
    )


class MyRETaskModuleWithWrongDocConverter(
    TaskModuleWithDocumentConverter, RETextClassificationWithIndicesTaskModule
):
    @property
    def document_type(self) -> Optional[Type[Document]]:
        return TestDocument

    def _convert_document(self, document: DocumentType) -> ConvertedDocumentType:
        result = TextDocumentWithLabeledSpansAndBinaryRelations(text="dummy")
        result.metadata["original_document"] = None
        return result

    def _integrate_predictions_from_converted_document(
        self, document: DocumentType, converted_document: ConvertedDocumentType
    ) -> None:
        pass


def test_wrong_doc_converter(documents):
    taskmodule = MyRETaskModuleWithWrongDocConverter(tokenizer_name_or_path="bert-base-cased")
    taskmodule.prepare(documents)
    with pytest.raises(ValueError) as e:
        taskmodule.encode(documents, encode_target=True)
    assert (
        str(e.value)
        == "metadata of converted_document has already and entry 'original_document', "
        "this is not allowed. Please adjust "
        "'MyRETaskModuleWithWrongDocConverter._convert_document()' to produce "
        "documents without that key in metadata."
    )
