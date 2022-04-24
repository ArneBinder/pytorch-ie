# import pytest

# from pytorch_ie.utils.span import (
#     convert_span_annotations_to_tag_sequence,
#     get_char_to_token_mapper,
#     has_overlap,
# )
# from tests.fixtures.document import get_doc1, get_doc2, get_doc3


# @pytest.fixture
# def documents():
#     doc_kwargs = dict(
#         assert_span_text=True,
#     )
#     documents = [get_doc1(**doc_kwargs), get_doc2(**doc_kwargs), get_doc3(**doc_kwargs)]
#     return documents


# @pytest.mark.skip
# def test_get_char_to_token_mapper():
#     # TODO: implement!
#     pass


# @pytest.mark.skip
# def test_get_special_token_mask():
#     # TODO: implement!
#     pass


# def test_convert_span_annotations_to_tag_sequence(documents):
#     doc = documents[0]
#     entities = doc.annotations.spans["entities"]
#     assert len(entities) == 3
#     char_to_token_mapping = {
#         0: 1,
#         1: 1,
#         2: 1,
#         3: 1,
#         5: 2,
#         6: 2,
#         7: 2,
#         8: 2,
#         9: 2,
#         11: 3,
#         12: 3,
#         14: 4,
#         15: 4,
#         16: 4,
#         17: 4,
#         18: 4,
#         19: 4,
#         20: 5,
#         22: 6,
#         23: 6,
#         24: 6,
#         25: 6,
#         27: 7,
#         28: 7,
#         30: 8,
#         31: 8,
#         33: 9,
#         34: 9,
#         35: 9,
#         36: 9,
#         37: 9,
#         38: 9,
#         39: 9,
#         40: 9,
#         42: 10,
#         43: 10,
#         44: 10,
#         45: 10,
#         46: 10,
#         48: 11,
#         49: 11,
#         50: 11,
#         51: 11,
#     }
#     char_to_token_mapper = get_char_to_token_mapper(
#         char_to_token_mapping=char_to_token_mapping,
#     )
#     special_tokens_mask = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#     tag_sequence = convert_span_annotations_to_tag_sequence(
#         spans=entities,
#         special_tokens_mask=special_tokens_mask,
#         char_to_token_mapper=char_to_token_mapper,
#     )
#     assert tag_sequence == [
#         None,
#         "B-person",
#         "O",
#         "O",
#         "B-city",
#         "O",
#         "O",
#         "O",
#         "O",
#         "O",
#         "O",
#         "B-person",
#         None,
#     ]

#     doc = documents[1]
#     entities = doc.annotations.spans["entities"]
#     assert len(entities) == 2
#     char_to_token_mapping = {
#         0: 1,
#         1: 1,
#         2: 1,
#         3: 1,
#         4: 1,
#         5: 1,
#         6: 1,
#         8: 2,
#         9: 2,
#         11: 3,
#         13: 4,
#         14: 4,
#         15: 4,
#         16: 4,
#         17: 4,
#         19: 5,
#         20: 5,
#         21: 5,
#         22: 5,
#         23: 6,
#         25: 7,
#         26: 7,
#         27: 7,
#         28: 7,
#         29: 7,
#         31: 8,
#         32: 8,
#         33: 9,
#         34: 9,
#         35: 9,
#         36: 10,
#         38: 11,
#         39: 11,
#         41: 12,
#         42: 12,
#         43: 12,
#         45: 13,
#         46: 13,
#         47: 13,
#         48: 13,
#         49: 14,
#         50: 15,
#         52: 16,
#         53: 16,
#         54: 16,
#         55: 16,
#         56: 16,
#         57: 17,
#     }
#     char_to_token_mapper = get_char_to_token_mapper(
#         char_to_token_mapping=char_to_token_mapping,
#     )
#     special_tokens_mask = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#     tag_sequence = convert_span_annotations_to_tag_sequence(
#         spans=entities,
#         special_tokens_mask=special_tokens_mask,
#         char_to_token_mapper=char_to_token_mapper,
#     )
#     assert tag_sequence == [
#         None,
#         "B-city",
#         "O",
#         "O",
#         "O",
#         "O",
#         "O",
#         "B-person",
#         "I-person",
#         "I-person",
#         "I-person",
#         "O",
#         "O",
#         "O",
#         "O",
#         "O",
#         "O",
#         "O",
#         None,
#     ]

#     doc = documents[2]
#     entities = doc.annotations.spans["entities"]
#     assert len(entities) == 2
#     char_to_token_mapping = {
#         0: 1,
#         1: 1,
#         2: 1,
#         3: 1,
#         5: 2,
#         6: 2,
#         7: 2,
#         8: 2,
#         9: 2,
#         10: 2,
#         12: 3,
#         13: 3,
#         14: 3,
#         15: 3,
#         16: 3,
#         18: 4,
#         19: 4,
#         20: 4,
#         21: 4,
#         23: 5,
#         24: 5,
#         26: 6,
#         27: 6,
#         28: 6,
#         29: 6,
#         30: 6,
#         31: 6,
#         32: 7,
#     }
#     char_to_token_mapper = get_char_to_token_mapper(
#         char_to_token_mapping=char_to_token_mapping,
#     )
#     special_tokens_mask = [1, 0, 0, 0, 0, 0, 0, 0, 1]
#     tag_sequence = convert_span_annotations_to_tag_sequence(
#         spans=entities,
#         special_tokens_mask=special_tokens_mask,
#         char_to_token_mapper=char_to_token_mapper,
#     )
#     assert tag_sequence == [None, "B-person", "O", "O", "O", "O", "B-city", "O", None]


# def test_convert_span_annotations_to_tag_sequence_with_partition(documents):
#     doc = documents[0]
#     entities = doc.annotations.spans["entities"]
#     assert len(entities) == 3
#     partitions = doc.annotations.spans["sentences"]
#     assert len(partitions) == 1
#     partition = partitions[0]
#     char_to_token_mapping = {
#         0: 1,
#         1: 1,
#         2: 1,
#         3: 1,
#         5: 2,
#         6: 2,
#         7: 2,
#         8: 2,
#         9: 2,
#         11: 3,
#         12: 3,
#         14: 4,
#         15: 4,
#         16: 4,
#         17: 4,
#         18: 4,
#         19: 4,
#         20: 5,
#     }
#     special_tokens_mask = [1, 0, 0, 0, 0, 0, 1]
#     char_to_token_mapper = get_char_to_token_mapper(
#         char_to_token_mapping=char_to_token_mapping,
#     )
#     tag_sequence = convert_span_annotations_to_tag_sequence(
#         spans=entities,
#         special_tokens_mask=special_tokens_mask,
#         char_to_token_mapper=char_to_token_mapper,
#         partition=partition,
#     )
#     assert tag_sequence == [None, "B-person", "O", "O", "B-city", "O", None]

#     doc = documents[1]
#     entities = doc.annotations.spans["entities"]
#     assert len(entities) == 2
#     partitions = doc.annotations.spans["sentences"]
#     assert len(partitions) == 2
#     partition = partitions[0]
#     char_to_token_mapping = {
#         0: 1,
#         1: 1,
#         2: 1,
#         3: 1,
#         4: 1,
#         5: 1,
#         6: 1,
#         8: 2,
#         9: 2,
#         11: 3,
#         13: 4,
#         14: 4,
#         15: 4,
#         16: 4,
#         17: 4,
#         19: 5,
#         20: 5,
#         21: 5,
#         22: 5,
#         23: 6,
#     }
#     special_tokens_mask = [1, 0, 0, 0, 0, 0, 0, 1]
#     char_to_token_mapper = get_char_to_token_mapper(
#         char_to_token_mapping=char_to_token_mapping,
#     )
#     tag_sequence = convert_span_annotations_to_tag_sequence(
#         spans=entities,
#         special_tokens_mask=special_tokens_mask,
#         char_to_token_mapper=char_to_token_mapper,
#         partition=partition,
#     )
#     assert tag_sequence == [None, "B-city", "O", "O", "O", "O", "O", None]
#     partition = partitions[1]
#     char_to_token_mapping = {
#         0: 1,
#         1: 1,
#         2: 1,
#         3: 1,
#         4: 1,
#         6: 2,
#         7: 2,
#         8: 3,
#         9: 3,
#         10: 3,
#         11: 4,
#         13: 5,
#         14: 5,
#         16: 6,
#         17: 6,
#         18: 6,
#         20: 7,
#         21: 7,
#         22: 7,
#         23: 7,
#         24: 8,
#         25: 9,
#         27: 10,
#         28: 10,
#         29: 10,
#         30: 10,
#         31: 10,
#         32: 11,
#     }
#     special_tokens_mask = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#     char_to_token_mapper = get_char_to_token_mapper(
#         char_to_token_mapping=char_to_token_mapping,
#     )
#     tag_sequence = convert_span_annotations_to_tag_sequence(
#         spans=entities,
#         special_tokens_mask=special_tokens_mask,
#         char_to_token_mapper=char_to_token_mapper,
#         partition=partition,
#     )
#     assert tag_sequence == [
#         None,
#         "B-person",
#         "I-person",
#         "I-person",
#         "I-person",
#         "O",
#         "O",
#         "O",
#         "O",
#         "O",
#         "O",
#         "O",
#         None,
#     ]

#     doc = documents[2]
#     entities = doc.annotations.spans["entities"]
#     assert len(entities) == 2
#     partitions = doc.annotations.spans["sentences"]
#     assert len(partitions) == 1
#     partition = partitions[0]
#     char_to_token_mapping = {
#         0: 1,
#         1: 1,
#         2: 1,
#         3: 1,
#         5: 2,
#         6: 2,
#         7: 2,
#         8: 2,
#         9: 2,
#         10: 2,
#         12: 3,
#         13: 3,
#         14: 3,
#         15: 3,
#         16: 3,
#         18: 4,
#         19: 4,
#         20: 4,
#         21: 4,
#         23: 5,
#         24: 5,
#         26: 6,
#         27: 6,
#         28: 6,
#         29: 6,
#         30: 6,
#         31: 6,
#         32: 7,
#     }
#     special_tokens_mask = [1, 0, 0, 0, 0, 0, 0, 0, 1]
#     char_to_token_mapper = get_char_to_token_mapper(
#         char_to_token_mapping=char_to_token_mapping,
#     )
#     tag_sequence = convert_span_annotations_to_tag_sequence(
#         spans=entities,
#         special_tokens_mask=special_tokens_mask,
#         char_to_token_mapper=char_to_token_mapper,
#         partition=partition,
#     )
#     assert tag_sequence == [None, "B-person", "O", "O", "O", "O", "B-city", "O", None]


# def test_has_overlap():
#     # no overlap - not touching
#     assert not has_overlap((3, 5), (6, 10))
#     assert not has_overlap((6, 10), (3, 5))

#     # no overlap - touching
#     assert not has_overlap((5, 10), (3, 5))
#     assert not has_overlap((3, 5), (5, 10))

#     # partly overlap
#     assert has_overlap((3, 5), (4, 10))
#     assert has_overlap((4, 10), (3, 5))

#     # partly overlap - same start
#     assert has_overlap((3, 5), (3, 10))
#     assert has_overlap((3, 10), (3, 5))

#     # partly overlap - same end
#     assert has_overlap((3, 5), (2, 5))
#     assert has_overlap((2, 5), (3, 5))

#     # total overlap (containing)
#     assert has_overlap((3, 5), (2, 10))
#     assert has_overlap((2, 10), (3, 5))
