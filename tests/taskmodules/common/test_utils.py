import torch

from pytorch_ie.taskmodules.common.utils import get_first_occurrence_index


def test_get_first_occurrence_index():
    tensor: torch.LongTensor = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1],  # 1
            [0, 0, 1, 1, 1, 1],  # 2
            [0, 1, 1, 0, 0, 1],  # 1
            [1, 1, 1, 1, 1, 1],  # 0
            [0, 0, 0, 0, 0, 0],  # 6 (=size of input) because no 1s at all
        ]
    ).to(torch.long)
    indices = get_first_occurrence_index(tensor, 1)
    torch.testing.assert_close(indices, torch.tensor([1, 2, 1, 0, 6]))
