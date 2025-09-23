import logging
from copy import copy
from typing import Any, Dict, List, Optional, Tuple

from torch import Tensor, nn

logger = logging.getLogger(__name__)

RNN_TYPE2CLASS = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
ACTIVATION_TYPE2CLASS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}


class RNNWrapper(nn.Module):
    def __init__(self, rnn: nn.Module):
        super().__init__()
        self.rnn = rnn

    def forward(self, *args, **kwargs) -> Tensor:
        return self.rnn(*args, **kwargs)[0]

    @property
    def output_size(self) -> int:
        if self.rnn.bidirectional:
            return self.rnn.hidden_size * 2
        else:
            return self.rnn.hidden_size


def build_seq2seq_encoder(
    config: Dict[str, Any], input_size: int
) -> Tuple[Optional[nn.Module], int]:
    # copy the config to avoid side effects
    config = copy(config)
    seq2seq_encoder_type = config.pop("type", None)
    seq2seq_encoder: Optional[nn.Module]
    if seq2seq_encoder_type is None:
        logger.warning(
            f"seq2seq_encoder_type is not specified in the seq2seq_encoder: {config}. "
            f"Do not build this seq2seq_encoder."
        )
        return None, input_size

    if seq2seq_encoder_type == "sequential":
        modules: List[nn.Module] = []
        output_size = input_size
        for key, subconfig in config.items():
            module, output_size = build_seq2seq_encoder(subconfig, input_size)
            if module is not None:
                modules.append(module)
            input_size = output_size

        seq2seq_encoder = nn.Sequential(*modules)
    elif seq2seq_encoder_type in RNN_TYPE2CLASS:
        rnn_class = RNN_TYPE2CLASS[seq2seq_encoder_type]
        seq2seq_encoder = RNNWrapper(rnn_class(input_size=input_size, batch_first=True, **config))
        output_size = seq2seq_encoder.output_size
    elif seq2seq_encoder_type == "linear":
        seq2seq_encoder = nn.Linear(in_features=input_size, **config)
        output_size = seq2seq_encoder.out_features
    elif seq2seq_encoder_type in ACTIVATION_TYPE2CLASS:
        activation_class = ACTIVATION_TYPE2CLASS[seq2seq_encoder_type]
        seq2seq_encoder = activation_class(**config)
        output_size = input_size
    elif seq2seq_encoder_type == "dropout":
        seq2seq_encoder = nn.Dropout(**config)
        output_size = input_size
    elif seq2seq_encoder_type == "none":
        seq2seq_encoder = None
        output_size = input_size
    else:
        raise ValueError(f"Unknown seq2seq_encoder_type: {seq2seq_encoder_type}")

    return seq2seq_encoder, output_size
