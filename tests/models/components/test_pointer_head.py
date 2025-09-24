import pytest
import torch
from torch import nn

from pytorch_ie.models.components.pointer_head import PointerHead


def get_pointer_head(num_embeddings=120, embedding_dim=3, eos_id=1, pad_id=2, **kwargs):
    torch.manual_seed(42)
    return PointerHead(
        embeddings=nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim),
        # bos, eos, pad, 3 x label ids
        target_token_ids=[100, 101, 102, 110, 111, 112],
        bos_id=0,  # -> 100
        eos_id=eos_id,  # 1 (default) -> 101
        pad_id=pad_id,  # 2 (default) -> 102
        embedding_weight_mapping={
            "110": [20, 21],
            "111": [30],
        },
        use_encoder_mlp=True,
        use_constraints_encoder_mlp=True,
        **kwargs,
    )


def test_get_pointer_head():
    pointer_head = get_pointer_head()
    assert pointer_head is not None
    assert not pointer_head.use_prepared_position_ids


def test_set_embeddings():
    pointer_head = get_pointer_head()
    original_embeddings = pointer_head.embeddings
    new_embeddings = nn.Embedding(
        original_embeddings.num_embeddings, original_embeddings.embedding_dim
    )
    pointer_head.set_embeddings(new_embeddings)
    assert pointer_head.embeddings is not None
    assert pointer_head.embeddings != original_embeddings
    assert pointer_head.embeddings == new_embeddings


def test_overwrite_embeddings_with_mapping():
    pointer_head = get_pointer_head()
    original_embeddings_weight = pointer_head.embeddings.weight.clone()
    pointer_head.overwrite_embeddings_with_mapping()
    assert pointer_head.embeddings is not None
    assert not torch.equal(pointer_head.embeddings.weight, original_embeddings_weight)
    torch.testing.assert_close(
        pointer_head.embeddings.weight[110], original_embeddings_weight[[20, 21]].mean(dim=0)
    )
    torch.testing.assert_close(
        pointer_head.embeddings.weight[111], original_embeddings_weight[[30]].mean(dim=0)
    )


@pytest.mark.parametrize(
    "use_attention_mask",
    [True, False],
)
def test_prepare_decoder_input_ids(use_attention_mask):
    pointer_head = get_pointer_head()
    encoder_input_ids = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 0],
        ]
    ).to(torch.long)
    # we have 3 special tokens (bos, eos, pad) and 3 labels, so the offset is 6
    input_ids = torch.tensor(
        [
            # bos, offset (0=6-6), offset (1=7-6), label (3), label (4), offset (2=8-6)
            [0, 6, 7, 3, 4, 8],
            # bos, label (3), offset (3=9-6), eos, pad, pad
            [0, 3, 9, 1, 2, 2],
        ]
    ).to(torch.long)
    # this is the attention mask for the (decoder) input_ids, not the encoder_input_ids
    attention_mask = (
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0],
            ]
        ).to(torch.long)
        if use_attention_mask
        else None
    )

    prepared_decoder_input_ids = pointer_head.prepare_decoder_input_ids(
        input_ids=input_ids,
        encoder_input_ids=encoder_input_ids,
    )
    assert prepared_decoder_input_ids is not None
    assert prepared_decoder_input_ids.shape == input_ids.shape
    # to recap, the target2token_id mapping is (bos, eos, pad, 3 x label ids)
    torch.testing.assert_close(
        pointer_head.target2token_id, torch.tensor([100, 101, 102, 110, 111, 112])
    )
    # 3 labels + bos / pad
    assert pointer_head.pointer_offset == 6
    assert prepared_decoder_input_ids.tolist() == [
        # bos (0), offset (0=6-6), offset (1=7-6), label (3), label (4), offset (2=8-6)
        [100, 10, 11, 110, 111, 12],
        # bos (0), label (3), offset (3=9-6), eos (1), pad (2), pad (2)
        [100, 110, 23, 101, 102, 102],
    ]


def test_prepare_decoder_input_ids_out_of_bounds():
    pointer_head = get_pointer_head()
    # 3 labels + bos / pad
    assert pointer_head.pointer_offset == 6
    encoder_input_ids = torch.tensor(
        [
            [100, 101, 102],
        ]
    ).to(torch.long)
    input_ids = torch.tensor(
        [
            # 9 is out of bounds: > pointer_head.pointer_offset + len(encoder_input_ids)
            [0, 9],
        ]
    ).to(torch.long)

    with pytest.raises(ValueError) as excinfo:
        pointer_head.prepare_decoder_input_ids(
            input_ids=input_ids, encoder_input_ids=encoder_input_ids
        )
    assert str(excinfo.value) == (
        "encoder_input_ids_index.max() [3] must be smaller than encoder_input_length [3]!"
    )


@pytest.mark.parametrize(
    "decoder_position_id_mode",
    ["pattern", "pattern_with_increment", "mapping"],
)
def test_prepare_decoder_position_ids(decoder_position_id_mode):
    pointer_head = get_pointer_head(
        decoder_position_id_mode=decoder_position_id_mode,
        decoder_position_id_pattern=[0, 1, 1, 2],
        decoder_position_id_mapping={"default": 3, "vocab": 2, "bos": 0, "eos": 0, "pad": 1},
    )
    input_ids = torch.tensor(
        [
            # bos, offset (0=6-6), offset (1=7-6), label (3), label (4), offset (2=8-6)
            [0, 6, 7, 3, 4, 8],
            # bos, label (3), offset (3=9-6), eos, pad, pad
            [0, 3, 9, 1, 2, 2],
        ]
    ).to(torch.long)

    prepared_decoder_position_ids = pointer_head.prepare_decoder_position_ids(input_ids=input_ids)
    assert prepared_decoder_position_ids is not None
    assert prepared_decoder_position_ids.shape == input_ids.shape
    if decoder_position_id_mode == "pattern":
        assert prepared_decoder_position_ids.tolist() == [
            [0, 2, 3, 3, 4, 2],
            [0, 2, 3, 3, 1, 1],
        ]
    elif decoder_position_id_mode == "pattern_with_increment":
        # the position ids (except for position-bos=0 and position-pad=1) get increased by 3 per record
        # (which has length 4)
        assert prepared_decoder_position_ids.tolist() == [
            [0, 2, 3, 3, 4, 5],
            [0, 2, 3, 3, 1, 1],
        ]
    elif decoder_position_id_mode == "mapping":
        assert prepared_decoder_position_ids.tolist() == [
            [0, 3, 3, 2, 2, 3],
            [0, 2, 3, 0, 1, 1],
        ]
    else:
        raise ValueError(f"unknown decoder_position_id_mode={decoder_position_id_mode}")


def test_prepare_decoder_position_ids_unknown_mode():
    with pytest.raises(ValueError) as excinfo:
        get_pointer_head(decoder_position_id_mode="unknown")
    assert str(excinfo.value) == (
        'decoder_position_id_mode="unknown" is not supported, use one of "pattern", '
        '"pattern_with_increment", or "mapping"!'
    )


@pytest.mark.parametrize(
    "decoder_position_id_mode",
    ["pattern", "pattern_with_increment", "mapping"],
)
def test_prepare_decoder_position_ids_missing_parameter(decoder_position_id_mode):
    with pytest.raises(ValueError) as excinfo:
        get_pointer_head(decoder_position_id_mode=decoder_position_id_mode)
    if decoder_position_id_mode in ["pattern", "pattern_with_increment"]:
        assert (
            str(excinfo.value) == "decoder_position_id_pattern must be provided when using "
            'decoder_position_id_mode="pattern" or "pattern_with_increment"!'
        )
    elif decoder_position_id_mode == "mapping":
        assert (
            str(excinfo.value)
            == 'decoder_position_id_mode="mapping" requires decoder_position_id_mapping to be provided!'
        )
    else:
        raise ValueError(f"unknown decoder_position_id_mode={decoder_position_id_mode}")


def test_prepare_decoder_position_ids_with_wrong_mapping():
    input_ids = torch.tensor(
        [
            # bos, offset (0=6-6), offset (1=7-6), label (3), label (4), offset (2=8-6)
            [0, 6, 7, 3, 4, 8],
            # bos, label (3), offset (3=9-6), eos, pad, pad
            [0, 3, 9, 1, 2, 2],
        ]
    ).to(torch.long)

    # missing default
    pointer_head = get_pointer_head(
        decoder_position_id_mode="mapping",
        decoder_position_id_mapping={"vocab": 2, "bos": 0, "eos": 0, "pad": 1},
    )
    with pytest.raises(ValueError) as excinfo:
        pointer_head.prepare_decoder_position_ids(input_ids=input_ids)
    assert (
        str(excinfo.value)
        == "mapping must contain a default entry, but only contains ['vocab', 'bos', 'eos', 'pad']!"
    )

    # unknown key
    pointer_head = get_pointer_head(
        decoder_position_id_mode="mapping",
        decoder_position_id_mapping={
            "default": 3,
            "vocab": 2,
            "bos": 0,
            "eos": 0,
            "pad": 1,
            "unknown": 4,
        },
    )
    with pytest.raises(ValueError) as excinfo:
        pointer_head.prepare_decoder_position_ids(input_ids=input_ids)
    assert (
        str(excinfo.value) == "Mapping contains unknown key 'unknown' "
        "(mapping: {'default': 3, 'vocab': 2, 'bos': 0, 'eos': 0, 'pad': 1, 'unknown': 4})."
    )

    # multiple values for same input id
    pointer_head = get_pointer_head(
        # same id for eos and pad
        eos_id=1,
        pad_id=1,
        decoder_position_id_mode="mapping",
        decoder_position_id_mapping={
            "default": 3,
            "vocab": 2,
            "bos": 0,
            # different position ids for eos and pad, this is not allowed when eos and pad have the same id
            "eos": 0,
            "pad": 1,
        },
    )
    with pytest.raises(ValueError) as excinfo:
        pointer_head.prepare_decoder_position_ids(input_ids=input_ids)
    assert (
        str(excinfo.value)
        == "Can not set the position ids for 'pad' to 1 because it was already set to 0 by key 'eos'. "
        "Note that both, 'pad' and 'eos', have the same id (1), so their position_ids need to be "
        "also the same (position id mapping: {'default': 3, 'vocab': 2, 'bos': 0, 'eos': 0, 'pad': 1})."
    )


def test_prepare_decoder_inputs():
    pointer_head = get_pointer_head(
        decoder_position_id_mode="pattern", decoder_position_id_pattern=[0, 1, 1, 2]
    )
    encoder_input_ids = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 0],
        ]
    ).to(torch.long)
    input_ids = torch.tensor(
        [
            # bos, offset (0=6-6), offset (1=7-6), label (3), label (4), offset (2=8-6)
            [0, 6, 7, 3, 4, 8],
            # bos, label (3), offset (3=9-6), eos, pad, pad
            [0, 3, 9, 1, 2, 2],
        ]
    ).to(torch.long)

    decoder_inputs = pointer_head.prepare_decoder_inputs(
        input_ids=input_ids,
        encoder_input_ids=encoder_input_ids,
    )
    assert set(decoder_inputs.keys()) == {"input_ids", "position_ids"}
    assert decoder_inputs["input_ids"].shape == input_ids.shape
    assert decoder_inputs["position_ids"].shape == input_ids.shape
    # to recap, the target2token_id mapping is (bos, eos, pad, 3 x label ids)
    torch.testing.assert_close(
        pointer_head.target2token_id, torch.tensor([100, 101, 102, 110, 111, 112])
    )
    # 3 labels + bos / pad
    assert pointer_head.pointer_offset == 6
    assert decoder_inputs["input_ids"].tolist() == [
        # bos (0), offset (0=6-6), offset (1=7-6), label (3), label (4), offset (2=8-6)
        [100, 10, 11, 110, 111, 12],
        # bos (0), label (3), offset (3=9-6), eos (1), pad (2), pad (2)
        [100, 110, 23, 101, 102, 102],
    ]
    assert decoder_inputs["position_ids"].tolist() == [
        [0, 2, 3, 3, 4, 2],
        [0, 2, 3, 3, 1, 1],
    ]


def test_forward():
    pointer_head = get_pointer_head()
    # shape: (batch_size=2, input_sequence_length=5)
    encoder_input_ids = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 0],
        ]
    ).to(torch.long)
    encoder_attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ]
    ).to(torch.long)
    # shape: (batch_size=2, input_sequence_length=5, hidden_size=3)
    encoder_last_hidden_state = pointer_head.embeddings(encoder_input_ids)
    # shape: (batch_size=2, target_sequence_length=4)
    prepared_input_ids = torch.tensor(
        [
            # bos (0), offset (0=6-6), offset (1=7-6), label (3)
            [100, 10, 11, 110],
            # bos (0), label (3), offset (3=9-6), eos (1)
            [100, 110, 23, 101],
        ]
    ).to(torch.long)
    # shape: (batch_size=2, target_sequence_length=4)
    last_hidden_state = pointer_head.embeddings(prepared_input_ids)

    torch.manual_seed(42)
    logits, loss = pointer_head(
        encoder_input_ids=encoder_input_ids,
        encoder_attention_mask=encoder_attention_mask,
        encoder_last_hidden_state=encoder_last_hidden_state,
        last_hidden_state=last_hidden_state,
    )
    assert loss is None
    assert logits is not None
    # shape: (batch_size=2, target_sequence_length=4, num_targets+num_offsets=6+5==11)
    assert logits.shape == (2, 4, 11)
    torch.testing.assert_close(
        logits,
        torch.tensor(
            [
                [
                    [
                        -1.0000000138484279e24,
                        -0.9407045245170593,
                        -1.0000000138484279e24,
                        0.5535521507263184,
                        0.04295700043439865,
                        1.0467679500579834,
                        -1.110795497894287,
                        1.1652655601501465,
                        0.09444020688533783,
                        0.43052661418914795,
                        -1.0437036752700806,
                    ],
                    [
                        -1.0000000138484279e24,
                        1.1563994884490967,
                        -1.0000000138484279e24,
                        -0.8941665887832642,
                        -0.6862093806266785,
                        -1.154745101928711,
                        1.6984729766845703,
                        -1.3889904022216797,
                        -0.4076152741909027,
                        -1.0112841129302979,
                        0.9846026301383972,
                    ],
                    [
                        -1.0000000138484279e24,
                        -1.9377808570861816,
                        -1.0000000138484279e24,
                        2.437451124191284,
                        0.041493892669677734,
                        0.5383729338645935,
                        -1.5238577127456665,
                        1.6700562238693237,
                        -0.07231226563453674,
                        1.0911093950271606,
                        -0.9189060926437378,
                    ],
                    [
                        -1.0000000138484279e24,
                        -1.880744218826294,
                        -1.0000000138484279e24,
                        3.8719429969787598,
                        0.07287894189357758,
                        -1.3378281593322754,
                        -0.653921365737915,
                        0.783344566822052,
                        -0.3344290256500244,
                        1.3571363687515259,
                        0.5505899786949158,
                    ],
                ],
                [
                    [
                        -1.0000000138484279e24,
                        -0.9407045245170593,
                        -1.0000000138484279e24,
                        0.5535521507263184,
                        0.04295700043439865,
                        1.0467679500579834,
                        -1.0019789934158325,
                        0.6891120672225952,
                        -0.002076566219329834,
                        0.7561025619506836,
                        -1.0000000331813535e32,
                    ],
                    [
                        -1.0000000138484279e24,
                        -1.880744218826294,
                        -1.0000000138484279e24,
                        3.8719429969787598,
                        0.07287894189357758,
                        -1.3378281593322754,
                        -1.3875324726104736,
                        -2.124865770339966,
                        -2.559859275817871,
                        0.5425653457641602,
                        -1.0000000331813535e32,
                    ],
                    [
                        -1.0000000138484279e24,
                        -1.479057788848877,
                        -1.0000000138484279e24,
                        1.7857770919799805,
                        0.6723557114601135,
                        0.6378745436668396,
                        -2.262815475463867,
                        -0.1536862850189209,
                        -0.5338708758354187,
                        1.3628911972045898,
                        -1.0000000331813535e32,
                    ],
                    [
                        -1.0000000138484279e24,
                        1.1815755367279053,
                        -1.0000000138484279e24,
                        -1.880744218826294,
                        -0.10646091401576996,
                        0.1437276005744934,
                        1.0795626640319824,
                        0.6434042453765869,
                        1.0681594610214233,
                        -0.5814396142959595,
                        -1.0000000331813535e32,
                    ],
                ],
            ]
        ),
    )


@pytest.mark.parametrize(
    "with_constraints",
    [True, False],
)
def test_forward_with_labels(with_constraints):
    pointer_head = get_pointer_head(num_embeddings=300, embedding_dim=3)

    # shape: (batch_size=2, input_sequence_length=5)
    encoder_input_ids = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 0, 0],
        ]
    ).to(torch.long)
    encoder_attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ]
    ).to(torch.long)
    # shape: (batch_size=2, input_sequence_length=5, hidden_size=3)
    # encoder_last_hidden_state = pointer_head.embeddings(encoder_input_ids)
    # shape: (batch_size=2, target_sequence_length=4)
    prepared_input_ids = torch.tensor(
        [
            # bos (0), offset (0=6-6), offset (1=7-6), label (3)
            [100, 10, 11, 110],
            # bos (0), label (3), offset (3=9-6), eos (1)
            [100, 110, 23, 101],
        ]
    ).to(torch.long)
    # shape: (batch_size=2, target_sequence_length=4)
    # last_hidden_state = pointer_head.embeddings(prepared_input_ids)
    labels = torch.tensor(
        [
            # offset (0=6-6), offset (1=7-6), label (3), label (4)
            [6, 7, 3, 4],
            # label (3), offset (3=9-6), eos, pad, pad
            [3, 9, 1, 2],
        ]
    ).to(torch.long)
    decoder_attention_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ]
    ).to(torch.long)

    # shape: (batch_size=2, target_sequence_length=4, num_targets+num_offsets=6+5==11)
    constraints = (
        # recap: the target2token_id mapping is (bos, eos, pad, 3 x label ids)
        torch.tensor(
            [
                [
                    # allow all labels
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    # allow all offsets different from previous label id (3)
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    # allow all offsets different from previous label ids (3, 4)
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    # allow all offsets
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                ],
                [
                    # allow all labels
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    # allow all offsets
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    # allow all offsets equal or bigger than previous one (9) or eos
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    # allow only pad
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        ).to(torch.long)
    )

    torch.manual_seed(42)
    # shape: (batch_size=2, input_sequence_length=6, hidden_size=3)
    encoder_last_hidden_state = pointer_head.embeddings(encoder_input_ids)
    last_hidden_state = pointer_head.embeddings(prepared_input_ids)
    _, loss = pointer_head(
        encoder_input_ids=encoder_input_ids,
        encoder_attention_mask=encoder_attention_mask,
        encoder_last_hidden_state=encoder_last_hidden_state,
        last_hidden_state=last_hidden_state,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask,
        constraints=constraints if with_constraints else None,
    )
    assert loss is not None
    maybe_gradients = torch.autograd.grad(loss, pointer_head.parameters(), allow_unused=True)
    gradients = [g for g in maybe_gradients if g is not None]
    if not with_constraints:
        # embeddings.weight, 2 x (encoder_mlp.weight, encoder_mlp.bias)
        assert len(gradients) == 5
        # embeddings.weight (just check entries for special tokens and labels)
        torch.testing.assert_close(
            gradients[0][100:113],
            torch.tensor(
                [
                    [0.29642319679260254, 0.012336060404777527, 0.14099650084972382],
                    [0.015981415286660194, 0.17855659127235413, -0.21089009940624237],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [-0.8812153935432434, -0.43322375416755676, 0.07359108328819275],
                    [0.22255337238311768, 0.09604272246360779, 0.017692387104034424],
                    [-0.021408570930361748, -0.01747075282037258, 0.15882402658462524],
                ]
            ),
        )
        # first encoder_mlp.weight
        torch.testing.assert_close(
            gradients[1],
            torch.tensor(
                [
                    [6.044770998414606e-05, -0.001140016596764326, 0.0007320810691453516],
                    [0.014351745136082172, 0.01521987747400999, -0.028653975576162338],
                    [0.011420723050832748, 0.0070406426675617695, -0.030101824551820755],
                ]
            ),
        )
        # first encoder_mlp.bias
        torch.testing.assert_close(
            gradients[2],
            torch.tensor([-0.0006180311902426183, -0.023118967190384865, -0.024205176159739494]),
        )
        # second encoder_mlp.weight
        torch.testing.assert_close(
            gradients[3],
            torch.tensor(
                [
                    [-0.0005463349516503513, -0.016356423497200012, 0.01958528161048889],
                    [-0.0005303063080646098, -0.029644077643752098, -0.1391362100839615],
                    [0.0028533015865832567, 0.08096987009048462, 0.28279614448547363],
                ]
            ),
        )
        # second encoder_mlp.bias
        torch.testing.assert_close(
            gradients[4],
            torch.tensor([-0.030467912554740906, -0.045307278633117676, 0.06145985424518585]),
        )
    else:
        # embeddings.weight, 2 x (encoder_mlp.weight, encoder_mlp.bias), 2 x (constraints_encoder_mlp.weight, constraints_encoder_mlp.bias)
        assert len(gradients) == 9
        # embeddings.weight (just check entries for special tokens and labels)
        torch.testing.assert_close(
            gradients[0][100:113],
            torch.tensor(
                [
                    [0.2915953993797302, 0.009700030088424683, 0.1484404355287552],
                    [0.02216985821723938, 0.15251068770885468, -0.21624334156513214],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [-0.8804605007171631, -0.4300656318664551, 0.0664108395576477],
                    [0.21543428301811218, 0.093157559633255, 0.013825103640556335],
                    [-0.021408570930361748, -0.01747075282037258, 0.15882402658462524],
                ]
            ),
        )
        # first encoder_mlp.weight
        torch.testing.assert_close(
            gradients[1],
            torch.tensor(
                [
                    [-0.0003244421095587313, 0.006118832156062126, -0.003929311875253916],
                    [0.013681752607226372, 0.013532182201743126, -0.027564184740185738],
                    [0.012365758419036865, 0.00791379064321518, -0.02969365194439888],
                ]
            ),
        )
        # first encoder_mlp.bias
        torch.testing.assert_close(
            gradients[2],
            torch.tensor([0.003317170077934861, -0.021803036332130432, -0.023893579840660095]),
        )
        # second encoder_mlp.weight
        torch.testing.assert_close(
            gradients[3],
            torch.tensor(
                [
                    [-0.004014550242573023, -0.018573174253106117, 0.019694898277521133],
                    [-0.0019358742283657193, -0.030542463064193726, -0.13909178972244263],
                    [0.0009692738531157374, 0.0797656774520874, 0.28285568952560425],
                ]
            ),
        )
        # second encoder_mlp.bias
        torch.testing.assert_close(
            gradients[4],
            torch.tensor([-0.046919066458940506, -0.05197446048259735, 0.05252313241362572]),
        )
        # first constraints_encoder_mlp.weight
        torch.testing.assert_close(
            gradients[5],
            torch.tensor(
                [
                    [0.010755524039268494, -0.009512078016996384, -0.007983260788023472],
                    [0.004236628767102957, 0.002073169220238924, -0.0010695274686440825],
                    [-0.008700753562152386, -0.00425766222178936, 0.002196485875174403],
                ]
            ),
        )
        # first constraints_encoder_mlp.bias
        torch.testing.assert_close(
            gradients[6],
            torch.tensor([0.05254765599966049, -0.0024578727316111326, 0.005047726910561323]),
        )
        # second constraints_encoder_mlp.weight
        torch.testing.assert_close(
            gradients[7],
            torch.tensor(
                [
                    [0.004190368112176657, -0.01078515499830246, -0.015312351286411285],
                    [0.001505501102656126, -0.006679146084934473, -0.009482797235250473],
                    [0.02189277485013008, -0.010388202033936977, -0.014748772606253624],
                ]
            ),
        )
        # second constraints_encoder_mlp.bias
        torch.testing.assert_close(
            gradients[8],
            torch.tensor([0.016296036541461945, -0.00018996000289916992, 0.05888192355632782]),
        )
