# import copy
# import os

# import pytest
# import torch

# from pytorch_ie.taskmodules import TransformerRETextClassificationTaskModule
# from pytorch_ie.taskmodules.transformer_re_text_classification import _enumerate_entity_pairs


# @pytest.fixture(scope="module")
# def taskmodule():
#     tokenizer_name_or_path = "bert-base-cased"
#     taskmodule = TransformerRETextClassificationTaskModule(
#         tokenizer_name_or_path=tokenizer_name_or_path,
#     )
#     return taskmodule


# @pytest.fixture
# def prepared_taskmodule(taskmodule, documents):
#     taskmodule.prepare(documents)
#     return taskmodule


# @pytest.fixture(scope="module", params=[False, True])
# def taskmodule_optional_marker(request):
#     tokenizer_name_or_path = "bert-base-cased"
#     taskmodule = TransformerRETextClassificationTaskModule(
#         tokenizer_name_or_path=tokenizer_name_or_path, add_type_to_marker=request.param
#     )
#     return taskmodule


# @pytest.fixture
# def prepared_taskmodule_optional_marker(taskmodule_optional_marker, documents):
#     taskmodule_optional_marker.prepare(documents)
#     return taskmodule_optional_marker


# @pytest.fixture
# def model_output():
#     return {
#         "logits": torch.tensor(
#             [
#                 [-1.6924, 9.5473, -1.9625],
#                 [-0.9995, -2.5705, 10.0095],
#             ]
#         ),
#     }


# def test_prepare(taskmodule_optional_marker, documents):
#     assert not taskmodule_optional_marker.is_prepared()
#     taskmodule_optional_marker.prepare(documents)
#     assert taskmodule_optional_marker.is_prepared()
#     assert set(taskmodule_optional_marker.label_to_id.keys()) == {
#         "no_relation",
#         "mayor_of",
#         "lives_in",
#     }
#     assert taskmodule_optional_marker.label_to_id["no_relation"] == 0
#     if taskmodule_optional_marker.add_type_to_marker:
#         assert taskmodule_optional_marker.argument_markers == {
#             ("head", "end", "city"): "[/H:city]",
#             ("head", "end", "person"): "[/H:person]",
#             ("tail", "end", "city"): "[/T:city]",
#             ("tail", "end", "person"): "[/T:person]",
#             ("head", "start", "city"): "[H:city]",
#             ("head", "start", "person"): "[H:person]",
#             ("tail", "start", "city"): "[T:city]",
#             ("tail", "start", "person"): "[T:person]",
#         }
#     else:
#         assert taskmodule_optional_marker.argument_markers == {
#             ("head", "end"): "[/H]",
#             ("tail", "end"): "[/T]",
#             ("head", "start"): "[H]",
#             ("tail", "start"): "[T]",
#         }


# def test_config(prepared_taskmodule_optional_marker):
#     config = prepared_taskmodule_optional_marker._config()
#     assert config["taskmodule_type"] == "TransformerRETextClassificationTaskModule"
#     assert "label_to_id" in config
#     assert set(config["label_to_id"]) == {"no_relation", "mayor_of", "lives_in"}
#     if prepared_taskmodule_optional_marker.add_type_to_marker:
#         assert set(config["entity_labels"]) == {"person", "city"}
#     else:
#         assert config["entity_labels"] == []


# @pytest.mark.parametrize("is_training", [False, True])
# def test_encode_input(prepared_taskmodule_optional_marker, documents, is_training):
#     (
#         input_encoding,
#         metadata,
#         new_documents,
#     ) = prepared_taskmodule_optional_marker.encode_input(documents, is_training=is_training)
#     tokens = [
#         prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
#         for encoding in input_encoding
#     ]
#     if is_training:
#         assert len(input_encoding) == 2
#         assert new_documents is not None
#         assert len(new_documents) == 2
#         assert new_documents[0].text == DOC1_TEXT
#         assert new_documents[1].text == DOC2_TEXT

#         if prepared_taskmodule_optional_marker.add_type_to_marker:
#             assert tokens[0] == [
#                 "[CLS]",
#                 "[H:person]",
#                 "Jane",
#                 "[/H:person]",
#                 "lives",
#                 "in",
#                 "[T:city]",
#                 "Berlin",
#                 "[/T:city]",
#                 ".",
#                 "this",
#                 "is",
#                 "no",
#                 "sentence",
#                 "about",
#                 "Karl",
#                 "[SEP]",
#             ]
#             assert tokens[1] == [
#                 "[CLS]",
#                 "[T:city]",
#                 "Seattle",
#                 "[/T:city]",
#                 "is",
#                 "a",
#                 "rainy",
#                 "city",
#                 ".",
#                 "[H:person]",
#                 "Jenny",
#                 "Du",
#                 "##rka",
#                 "##n",
#                 "[/H:person]",
#                 "is",
#                 "the",
#                 "city",
#                 "'",
#                 "s",
#                 "mayor",
#                 ".",
#                 "[SEP]",
#             ]
#         else:
#             assert tokens[0] == [
#                 "[CLS]",
#                 "[H]",
#                 "Jane",
#                 "[/H]",
#                 "lives",
#                 "in",
#                 "[T]",
#                 "Berlin",
#                 "[/T]",
#                 ".",
#                 "this",
#                 "is",
#                 "no",
#                 "sentence",
#                 "about",
#                 "Karl",
#                 "[SEP]",
#             ]
#             assert tokens[1] == [
#                 "[CLS]",
#                 "[T]",
#                 "Seattle",
#                 "[/T]",
#                 "is",
#                 "a",
#                 "rainy",
#                 "city",
#                 ".",
#                 "[H]",
#                 "Jenny",
#                 "Du",
#                 "##rka",
#                 "##n",
#                 "[/H]",
#                 "is",
#                 "the",
#                 "city",
#                 "'",
#                 "s",
#                 "mayor",
#                 ".",
#                 "[SEP]",
#             ]
#     else:
#         assert len(input_encoding) == 10
#         assert new_documents is not None
#         assert len(new_documents) == 10
#         assert new_documents[0].text == DOC1_TEXT
#         assert new_documents[6].text == DOC2_TEXT
#         assert new_documents[8].text == DOC3_TEXT

#         if prepared_taskmodule_optional_marker.add_type_to_marker:
#             # just check the first two entries
#             assert tokens[0] == [
#                 "[CLS]",
#                 "[H:person]",
#                 "Jane",
#                 "[/H:person]",
#                 "lives",
#                 "in",
#                 "[T:city]",
#                 "Berlin",
#                 "[/T:city]",
#                 ".",
#                 "this",
#                 "is",
#                 "no",
#                 "sentence",
#                 "about",
#                 "Karl",
#                 "[SEP]",
#             ]
#             assert tokens[1] == [
#                 "[CLS]",
#                 "[H:person]",
#                 "Jane",
#                 "[/H:person]",
#                 "lives",
#                 "in",
#                 "Berlin",
#                 ".",
#                 "this",
#                 "is",
#                 "no",
#                 "sentence",
#                 "about",
#                 "[T:person]",
#                 "Karl",
#                 "[/T:person]",
#                 "[SEP]",
#             ]
#         else:
#             # just check the first two entries
#             assert tokens[0] == [
#                 "[CLS]",
#                 "[H]",
#                 "Jane",
#                 "[/H]",
#                 "lives",
#                 "in",
#                 "[T]",
#                 "Berlin",
#                 "[/T]",
#                 ".",
#                 "this",
#                 "is",
#                 "no",
#                 "sentence",
#                 "about",
#                 "Karl",
#                 "[SEP]",
#             ]
#             assert tokens[1] == [
#                 "[CLS]",
#                 "[H]",
#                 "Jane",
#                 "[/H]",
#                 "lives",
#                 "in",
#                 "Berlin",
#                 ".",
#                 "this",
#                 "is",
#                 "no",
#                 "sentence",
#                 "about",
#                 "[T]",
#                 "Karl",
#                 "[/T]",
#                 "[SEP]",
#             ]


# @pytest.mark.parametrize("encode_target", [False, True])
# def test_encode_target(prepared_taskmodule, documents, encode_target):
#     task_encodings = prepared_taskmodule.encode(documents, encode_target=encode_target)
#     if encode_target:
#         # in this case, the existing relations were taken into account
#         assert len(task_encodings) == 2
#         encoding = task_encodings[0]
#         assert encoding.has_target
#         target_labels = [prepared_taskmodule.id_to_label[_id] for _id in encoding.target]
#         assert target_labels == [DOC1_REL_LIVES_IN.label]

#         encoding = task_encodings[1]
#         assert encoding.has_target
#         target_labels = [prepared_taskmodule.id_to_label[_id] for _id in encoding.target]
#         assert target_labels == [DOC2_REL_MAYOR_OF.label]
#     else:
#         # all possible entity pairs are taken as candidates
#         assert len(task_encodings) == 10
#         assert [encoding.has_target for encoding in task_encodings] == [False] * len(
#             task_encodings
#         )


# @pytest.mark.parametrize("encode_target", [False, True])
# def test_encode(prepared_taskmodule_optional_marker, documents, encode_target):
#     # the code is actually tested in test_encode_input() and test_encode_target(). Here we only test assertions in encode().
#     task_encodings = prepared_taskmodule_optional_marker.encode(documents, encode_target=True)


# def test_encode_input_with_partitions(prepared_taskmodule_optional_marker, documents):
#     prepared_taskmodule_with_partitions = copy.deepcopy(prepared_taskmodule_optional_marker)
#     prepared_taskmodule_with_partitions.partition_annotation = "sentences"
#     input_encoding, metadata, new_documents = prepared_taskmodule_with_partitions.encode_input(
#         documents, is_training=True
#     )
#     assert len(input_encoding) == 1
#     assert new_documents is not None
#     assert len(new_documents) == 1
#     encoding = input_encoding[0]
#     if prepared_taskmodule_with_partitions.add_type_to_marker:
#         assert prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
#             encoding["input_ids"]
#         ) == [
#             "[CLS]",
#             "[H:person]",
#             "Jane",
#             "[/H:person]",
#             "lives",
#             "in",
#             "[T:city]",
#             "Berlin",
#             "[/T:city]",
#             ".",
#             "[SEP]",
#         ]
#     else:
#         assert prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(
#             encoding["input_ids"]
#         ) == [
#             "[CLS]",
#             "[H]",
#             "Jane",
#             "[/H]",
#             "lives",
#             "in",
#             "[T]",
#             "Berlin",
#             "[/T]",
#             ".",
#             "[SEP]",
#         ]


# def test_encode_with_windowing(prepared_taskmodule_optional_marker, documents):
#     prepared_taskmodule_with_windowing = copy.deepcopy(prepared_taskmodule_optional_marker)
#     prepared_taskmodule_with_windowing.max_window = 10
#     task_encodings = prepared_taskmodule_with_windowing.encode(documents, encode_target=False)
#     assert len(task_encodings) == 2

#     # just check the first entry
#     encoding = task_encodings[0]
#     document = documents[0]
#     assert encoding.document == document
#     assert "input_ids" in encoding.input
#     assert len(encoding.input["input_ids"]) <= prepared_taskmodule_with_windowing.max_window
#     if prepared_taskmodule_with_windowing.add_type_to_marker:
#         assert prepared_taskmodule_with_windowing.tokenizer.convert_ids_to_tokens(
#             encoding.input["input_ids"]
#         ) == [
#             "[CLS]",
#             "[H:person]",
#             "Jane",
#             "[/H:person]",
#             "lives",
#             "in",
#             "[T:city]",
#             "Berlin",
#             "[/T:city]",
#             "[SEP]",
#         ]
#     else:
#         assert prepared_taskmodule_with_windowing.tokenizer.convert_ids_to_tokens(
#             encoding.input["input_ids"]
#         ) == ["[CLS]", "[H]", "Jane", "[/H]", "lives", "in", "[T]", "Berlin", "[/T]", "[SEP]"]


# @pytest.mark.parametrize("encode_target", [False, True])
# def test_collate(prepared_taskmodule_optional_marker, documents, encode_target):
#     encodings = prepared_taskmodule_optional_marker.encode(documents, encode_target=encode_target)

#     if encode_target:
#         assert len(encodings) == 2
#         assert all([encoding.has_target for encoding in encodings])
#     else:
#         assert len(encodings) == 10
#         assert not any([encoding.has_target for encoding in encodings])

#     batch_encoding = prepared_taskmodule_optional_marker.collate(encodings)
#     inputs, targets = batch_encoding
#     assert "input_ids" in inputs
#     assert "attention_mask" in inputs
#     assert inputs["input_ids"].shape == inputs["attention_mask"].shape
#     tokens = [
#         prepared_taskmodule_optional_marker.tokenizer.convert_ids_to_tokens(input_ids)
#         for input_ids in inputs["input_ids"].tolist()
#     ]
#     if encode_target:
#         assert inputs["input_ids"].shape[0] == 2
#         if prepared_taskmodule_optional_marker.add_type_to_marker:
#             assert tokens[0] == [
#                 "[CLS]",
#                 "[H:person]",
#                 "Jane",
#                 "[/H:person]",
#                 "lives",
#                 "in",
#                 "[T:city]",
#                 "Berlin",
#                 "[/T:city]",
#                 ".",
#                 "this",
#                 "is",
#                 "no",
#                 "sentence",
#                 "about",
#                 "Karl",
#                 "[SEP]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#             ]
#             assert tokens[1] == [
#                 "[CLS]",
#                 "[T:city]",
#                 "Seattle",
#                 "[/T:city]",
#                 "is",
#                 "a",
#                 "rainy",
#                 "city",
#                 ".",
#                 "[H:person]",
#                 "Jenny",
#                 "Du",
#                 "##rka",
#                 "##n",
#                 "[/H:person]",
#                 "is",
#                 "the",
#                 "city",
#                 "'",
#                 "s",
#                 "mayor",
#                 ".",
#                 "[SEP]",
#             ]

#         else:
#             assert tokens[0] == [
#                 "[CLS]",
#                 "[H]",
#                 "Jane",
#                 "[/H]",
#                 "lives",
#                 "in",
#                 "[T]",
#                 "Berlin",
#                 "[/T]",
#                 ".",
#                 "this",
#                 "is",
#                 "no",
#                 "sentence",
#                 "about",
#                 "Karl",
#                 "[SEP]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#             ]
#             assert tokens[1] == [
#                 "[CLS]",
#                 "[T]",
#                 "Seattle",
#                 "[/T]",
#                 "is",
#                 "a",
#                 "rainy",
#                 "city",
#                 ".",
#                 "[H]",
#                 "Jenny",
#                 "Du",
#                 "##rka",
#                 "##n",
#                 "[/H]",
#                 "is",
#                 "the",
#                 "city",
#                 "'",
#                 "s",
#                 "mayor",
#                 ".",
#                 "[SEP]",
#             ]
#     else:
#         assert inputs["input_ids"].shape[0] == 10
#         if prepared_taskmodule_optional_marker.add_type_to_marker:
#             # just test the first entry
#             assert tokens[0] == [
#                 "[CLS]",
#                 "[H:person]",
#                 "Jane",
#                 "[/H:person]",
#                 "lives",
#                 "in",
#                 "[T:city]",
#                 "Berlin",
#                 "[/T:city]",
#                 ".",
#                 "this",
#                 "is",
#                 "no",
#                 "sentence",
#                 "about",
#                 "Karl",
#                 "[SEP]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#             ]
#         else:
#             # just test the first entry
#             assert tokens[0] == [
#                 "[CLS]",
#                 "[H]",
#                 "Jane",
#                 "[/H]",
#                 "lives",
#                 "in",
#                 "[T]",
#                 "Berlin",
#                 "[/T]",
#                 ".",
#                 "this",
#                 "is",
#                 "no",
#                 "sentence",
#                 "about",
#                 "Karl",
#                 "[SEP]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#                 "[PAD]",
#             ]

#     if encode_target:
#         assert targets.shape == (2,)
#         labels = [
#             prepared_taskmodule_optional_marker.id_to_label[target_id]
#             for target_id in targets.tolist()
#         ]
#         assert labels == [DOC1_REL_LIVES_IN.label, DOC2_REL_MAYOR_OF.label]
#     else:
#         assert targets is None


# def test_unbatch_output(prepared_taskmodule, model_output):
#     unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)

#     assert len(unbatched_outputs) == 2

#     unbatched_output1 = unbatched_outputs[0]
#     assert len(unbatched_output1["labels"]) == 1
#     assert len(unbatched_output1["probabilities"]) == 1
#     assert unbatched_output1["labels"][0] == DOC1_REL_LIVES_IN.label
#     assert unbatched_output1["probabilities"][0] == pytest.approx(0.9999768733978271)

#     unbatched_output2 = unbatched_outputs[1]
#     assert len(unbatched_output2["labels"]) == 1
#     assert len(unbatched_output2["probabilities"]) == 1
#     assert unbatched_output2["labels"][0] == DOC2_REL_MAYOR_OF.label
#     assert unbatched_output2["probabilities"][0] == pytest.approx(0.9999799728393555)


# @pytest.mark.parametrize("inplace", [False, True])
# def test_decode(prepared_taskmodule, documents, model_output, inplace):
#     encodings = prepared_taskmodule.encode(documents, encode_target=True)
#     unbatched_outputs = prepared_taskmodule.unbatch_output(model_output)
#     assert len(unbatched_outputs) == len(encodings)
#     decoded_documents = prepared_taskmodule.decode(
#         encodings=encodings,
#         decoded_outputs=unbatched_outputs,
#         inplace=inplace,
#         input_documents=documents,
#     )

#     assert len(decoded_documents) == 3
#     if inplace:
#         assert set(documents) == set(decoded_documents)
#     else:
#         for doc in decoded_documents:
#             assert doc not in set(documents)

#     predictions = decoded_documents[0].predictions.binary_relations["relations"]
#     assert len(predictions) == 1
#     prediction = predictions[0]
#     assert prediction.label == DOC1_REL_LIVES_IN.label
#     assert prediction.head == DOC1_ENTITY_JANE
#     assert prediction.tail == DOC1_ENTITY_BERLIN

#     predictions = decoded_documents[1].predictions.binary_relations["relations"]
#     assert len(predictions) == 1
#     prediction = predictions[0]
#     assert prediction.label == DOC2_REL_MAYOR_OF.label
#     assert prediction.head == DOC2_ENTITY_JENNY
#     assert prediction.tail == DOC2_ENTITY_SEATTLE

#     assert not decoded_documents[2].predictions.binary_relations.has_layer("relations")


# def test_save_load(tmp_path, prepared_taskmodule):
#     path = os.path.join(tmp_path, "taskmodule")
#     prepared_taskmodule.save_pretrained(path)
#     loaded_taskmodule = TransformerRETextClassificationTaskModule.from_pretrained(path)
#     assert loaded_taskmodule.is_prepared()
#     assert loaded_taskmodule.argument_markers == prepared_taskmodule.argument_markers


# def test_enumerate_entity_pairs(prepared_taskmodule, documents):
#     """
#     This should return all combinations of entities.
#     """
#     document = documents[0]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 3
#     assert entities[0] == DOC1_ENTITY_JANE
#     assert entities[1] == DOC1_ENTITY_BERLIN
#     assert entities[2] == DOC1_ENTITY_KARL

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC1_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,  # partition=partition, relations=relations,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 6

#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC1_ENTITY_JANE
#     assert tail == DOC1_ENTITY_BERLIN

#     head, tail = enumerated_entity_pairs[1]
#     assert head == DOC1_ENTITY_JANE
#     assert tail == DOC1_ENTITY_KARL

#     head, tail = enumerated_entity_pairs[2]
#     assert head == DOC1_ENTITY_BERLIN
#     assert tail == DOC1_ENTITY_JANE

#     head, tail = enumerated_entity_pairs[3]
#     assert head == DOC1_ENTITY_BERLIN
#     assert tail == DOC1_ENTITY_KARL

#     head, tail = enumerated_entity_pairs[4]
#     assert head == DOC1_ENTITY_KARL
#     assert tail == DOC1_ENTITY_JANE

#     head, tail = enumerated_entity_pairs[5]
#     assert head == DOC1_ENTITY_KARL
#     assert tail == DOC1_ENTITY_BERLIN

#     document = documents[1]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC2_ENTITY_SEATTLE
#     assert entities[1] == DOC2_ENTITY_JENNY

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC2_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,  # partition=partition, relations=relations,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 2

#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC2_ENTITY_SEATTLE
#     assert tail == DOC2_ENTITY_JENNY

#     head, tail = enumerated_entity_pairs[1]
#     assert head == DOC2_ENTITY_JENNY
#     assert tail == DOC2_ENTITY_SEATTLE

#     document = documents[2]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC3_ENTITY_KARL
#     assert entities[1] == DOC3_ENTITY_BERLIN

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC3_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,  # partition=partition, relations=relations,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 2

#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC3_ENTITY_KARL
#     assert tail == DOC3_ENTITY_BERLIN

#     head, tail = enumerated_entity_pairs[1]
#     assert head == DOC3_ENTITY_BERLIN
#     assert tail == DOC3_ENTITY_KARL


# def test_enumerate_entity_pairs_with_relations(prepared_taskmodule, documents):
#     """
#     This should return only combinations for which a relation exists.
#     """
#     document = documents[0]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 3
#     assert entities[0] == DOC1_ENTITY_JANE
#     assert entities[1] == DOC1_ENTITY_BERLIN
#     assert entities[2] == DOC1_ENTITY_KARL
#     relations = document.annotations.binary_relations["relations"]
#     assert len(relations) == 1
#     relation = relations[0]
#     assert relation == DOC1_REL_LIVES_IN
#     assert relation.head == DOC1_ENTITY_JANE
#     assert relation.tail == DOC1_ENTITY_BERLIN

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC1_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             relations=relations,
#             # partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 1
#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC1_ENTITY_JANE
#     assert tail == DOC1_ENTITY_BERLIN

#     document = documents[1]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC2_ENTITY_SEATTLE
#     assert entities[1] == DOC2_ENTITY_JENNY
#     relations = document.annotations.binary_relations["relations"]
#     assert len(relations) == 1
#     relation = relations[0]
#     assert relation == DOC2_REL_MAYOR_OF
#     assert relation.head == DOC2_ENTITY_JENNY
#     assert relation.tail == DOC2_ENTITY_SEATTLE

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC2_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             relations=relations,
#             # partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 1

#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC2_ENTITY_JENNY
#     assert tail == DOC2_ENTITY_SEATTLE

#     document = documents[2]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC3_ENTITY_KARL
#     assert entities[1] == DOC3_ENTITY_BERLIN
#     relations = document.annotations.binary_relations["relations"]
#     assert len(relations) == 0

#     encoding = prepared_taskmodule._encode_text(
#         document=document,  # partition=partition, add_special_tokens=add_special_tokens
#     )
#     assert (
#         prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"])
#         == DOC3_TOKENS
#     )

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             relations=relations,  # partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 0


# def test_enumerate_entity_pairs_with_partitions(prepared_taskmodule, documents):
#     """
#     This should return only combinations with entities in the same sentence.
#     """
#     document = documents[0]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 3
#     assert entities[0] == DOC1_ENTITY_JANE
#     assert entities[1] == DOC1_ENTITY_BERLIN
#     assert entities[2] == DOC1_ENTITY_KARL
#     sentences = document.annotations.spans["sentences"]
#     assert len(sentences) == 1
#     partition = sentences[0]
#     assert partition == DOC1_SENTENCE1

#     encoding = prepared_taskmodule._encode_text(
#         document=document,
#         partition=partition,  # add_special_tokens=add_special_tokens
#     )
#     assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
#         "[CLS]",
#         "Jane",
#         "lives",
#         "in",
#         "Berlin",
#         ".",
#         "[SEP]",
#     ]

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             # relations=relations,
#             partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 2
#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC1_ENTITY_JANE
#     assert tail == DOC1_ENTITY_BERLIN

#     head, tail = enumerated_entity_pairs[1]
#     assert head == DOC1_ENTITY_BERLIN
#     assert tail == DOC1_ENTITY_JANE

#     document = documents[1]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC2_ENTITY_SEATTLE
#     assert entities[1] == DOC2_ENTITY_JENNY
#     sentences = document.annotations.spans["sentences"]
#     assert len(sentences) == 2
#     partition = sentences[0]
#     assert partition == DOC2_SENTENCE1

#     encoding = prepared_taskmodule._encode_text(
#         document=document,
#         partition=partition,  # add_special_tokens=add_special_tokens
#     )
#     assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
#         "[CLS]",
#         "Seattle",
#         "is",
#         "a",
#         "rainy",
#         "city",
#         ".",
#         "[SEP]",
#     ]

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             # relations=relations,
#             partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 0

#     partition = sentences[1]
#     assert partition == DOC2_SENTENCE2

#     encoding = prepared_taskmodule._encode_text(
#         document=document,
#         partition=partition,  # add_special_tokens=add_special_tokens
#     )
#     assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
#         "[CLS]",
#         "Jenny",
#         "Du",
#         "##rka",
#         "##n",
#         "is",
#         "the",
#         "city",
#         "'",
#         "s",
#         "mayor",
#         ".",
#         "[SEP]",
#     ]

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             # relations=relations,
#             partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 0

#     document = documents[2]
#     entities = document.annotations.spans["entities"]
#     assert len(entities) == 2
#     assert entities[0] == DOC3_ENTITY_KARL
#     assert entities[1] == DOC3_ENTITY_BERLIN
#     sentences = document.annotations.spans["sentences"]
#     assert len(sentences) == 1
#     partition = sentences[0]
#     assert partition == DOC3_SENTENCE1

#     encoding = prepared_taskmodule._encode_text(
#         document=document,
#         partition=partition,  # add_special_tokens=add_special_tokens
#     )
#     assert prepared_taskmodule.tokenizer.convert_ids_to_tokens(encoding.data["input_ids"]) == [
#         "[CLS]",
#         "Karl",
#         "enjoys",
#         "sunny",
#         "days",
#         "in",
#         "Berlin",
#         ".",
#         "[SEP]",
#     ]

#     enumerated_entity_pairs = list(
#         _enumerate_entity_pairs(
#             entities=entities,
#             # relations=relations,
#             partition=partition,
#         )
#     )
#     assert len(enumerated_entity_pairs) == 2

#     head, tail = enumerated_entity_pairs[0]
#     assert head == DOC3_ENTITY_KARL
#     assert tail == DOC3_ENTITY_BERLIN

#     head, tail = enumerated_entity_pairs[1]
#     assert head == DOC3_ENTITY_BERLIN
#     assert tail == DOC3_ENTITY_KARL
