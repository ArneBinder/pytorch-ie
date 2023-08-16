import dataclasses

from transformers import AutoTokenizer, PreTrainedTokenizer

from pytorch_ie import text_based_document_to_token_based
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TokenBasedDocument


def test_text_based_document_to_token_based(documents):
    @dataclasses.dataclass
    class TokenizedTestDocument(TokenBasedDocument):
        sentences: AnnotationList[Span] = annotation_field(target="tokens")
        entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    assert len(documents) >= 3
    for i, doc in enumerate(documents[:3]):
        tokenized_text = tokenizer(doc.text, return_offsets_mapping=True)
        tokenized_doc = text_based_document_to_token_based(
            doc,
            tokens=tokenized_text.tokens(),
            text_span_layers=["sentences", "entities"],
            result_document_type=TokenizedTestDocument,
            token_offset_mapping=tokenized_text.offset_mapping,
            char_to_token=tokenized_text.char_to_token,
        )
        assert tokenized_doc is not None
        if i == 0:
            assert doc.id == "train_doc1"
            assert doc.text == "A single sentence."
            assert tokenized_doc.metadata["text"] == doc.text
            assert tokenized_doc.tokens == ("[CLS]", "A", "single", "sentence", ".", "[SEP]")
            assert len(tokenized_doc.sentences) == len(doc.sentences) == 1
            assert str(doc.sentences[0]) == "A single sentence."
            assert str(tokenized_doc.sentences[0]) == "('A', 'single', 'sentence', '.')"
            assert len(tokenized_doc.entities) == len(doc.entities) == 0
            assert len(tokenized_doc.relations) == len(doc.relations) == 0
        elif i == 1:
            assert doc.id == "train_doc2"
            assert doc.text == "Entity A works at B."
            assert tokenized_doc.metadata["text"] == doc.text
            assert tokenized_doc.tokens == (
                "[CLS]",
                "En",
                "##ti",
                "##ty",
                "A",
                "works",
                "at",
                "B",
                ".",
                "[SEP]",
            )
            assert len(tokenized_doc.sentences) == len(doc.sentences) == 1
            assert str(doc.sentences[0]) == "Entity A works at B."
            assert (
                str(tokenized_doc.sentences[0])
                == "('En', '##ti', '##ty', 'A', 'works', 'at', 'B', '.')"
            )
            assert len(tokenized_doc.entities) == len(doc.entities) == 2
            assert str(doc.entities[0]) == "Entity A"
            assert str(tokenized_doc.entities[0]) == "('En', '##ti', '##ty', 'A')"
            assert str(doc.entities[1]) == "B"
            assert str(tokenized_doc.entities[1]) == "('B',)"
            assert len(tokenized_doc.relations) == len(doc.relations) == 1
            assert doc.relations[0].head == doc.entities[0]
            assert tokenized_doc.relations[0].head == tokenized_doc.entities[0]
            assert doc.relations[0].tail == doc.entities[1]
            assert tokenized_doc.relations[0].tail == tokenized_doc.entities[1]
        elif i == 2:
            assert doc.id == "train_doc3"
            assert doc.text == "Entity C and D."
            assert tokenized_doc.metadata["text"] == doc.text
            assert tokenized_doc.tokens == (
                "[CLS]",
                "En",
                "##ti",
                "##ty",
                "C",
                "and",
                "D",
                ".",
                "[SEP]",
            )
            assert len(tokenized_doc.sentences) == len(doc.sentences) == 1
            assert str(doc.sentences[0]) == "Entity C and D."
            assert (
                str(tokenized_doc.sentences[0]) == "('En', '##ti', '##ty', 'C', 'and', 'D', '.')"
            )
            assert len(tokenized_doc.entities) == len(doc.entities) == 2
            assert str(doc.entities[0]) == "Entity C"
            assert str(tokenized_doc.entities[0]) == "('En', '##ti', '##ty', 'C')"
            assert str(doc.entities[1]) == "D"
            assert str(tokenized_doc.entities[1]) == "('D',)"
            assert len(tokenized_doc.relations) == len(doc.relations) == 0
        else:
            raise ValueError(f"Unexpected document: {doc.id}")
