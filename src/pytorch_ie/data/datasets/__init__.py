import pathlib
from typing import Dict, List, Union

from datasets import Split
from pytorch_ie import Document

HF_DATASETS_ROOT = pathlib.Path(__file__).parent / "hf_datasets"

PIEDatasetDict = Dict[Union[str, Split], List[Document]]
