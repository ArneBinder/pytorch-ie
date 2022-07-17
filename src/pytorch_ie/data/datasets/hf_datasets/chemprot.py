"""TODO: Add a description here."""


import json
import re

import datasets

_CITATION_CHEMPROT = """\
@article{article,
author = {Kringelum, Jens and Kjaerulff, Sonny and Brunak, Søren and Lund, Ole and Oprea, Tudor and Taboureau, Olivier},
year = {2016},
month = {02},
pages = {bav123},
title = {ChemProt-3.0: A global chemical biology diseases mapping},
volume = {2016},
journal = {Database},
doi = {10.1093/database/bav123}
}"""

# You can copy an official description
_DESCRIPTION = """\
ChemProt is a publicly available compilation of chemical-protein-disease annotation resources that enables the study
of systems pharmacology for a small molecule across multiple layers of complexity from molecular to clinical levels.
In this third version, ChemProt has been updated to more than 1.7 million compounds with 7.8 million bioactivity
measurements for 19 504 proteins.
"""

_HOMEPAGE = "http://potentia.cbs.dtu.dk/ChemProt/"

# TODO: Add the license for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here, currently pointing to preprocessed scibert files
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_DATA_URLs = {
    "train": "https://raw.githubusercontent.com/allenai/scibert/master/data/text_classification/chemprot/train.txt",
    "dev": "https://raw.githubusercontent.com/allenai/scibert/master/data/text_classification/chemprot/dev.txt",
    "test": "https://raw.githubusercontent.com/allenai/scibert/master/data/text_classification/chemprot/test.txt",
}

_CLASS_LABELS = [
    "ACTIVATOR",
    "AGONIST",
    "AGONIST-ACTIVATOR",
    "AGONIST-INHIBITOR",
    "ANTAGONIST",
    "DOWNREGULATOR",
    "INDIRECT-DOWNREGULATOR",
    "INDIRECT-UPREGULATOR",
    "INHIBITOR",
    "PRODUCT-OF",
    "SUBSTRATE",
    "SUBSTRATE_PRODUCT-OF",
    "UPREGULATOR",
]


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class ChemProt(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("3.0.0")  # type: ignore

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "head_start": datasets.Value("int32"),
                    "head_end": datasets.Value("int32"),
                    "tail_start": datasets.Value("int32"),
                    "tail_end": datasets.Value("int32"),
                    "label": datasets.ClassLabel(names=_CLASS_LABELS),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION_CHEMPROT,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        data_files = dl_manager.download_and_extract(_DATA_URLs)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files.get("train")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files.get("dev")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files.get("test")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                example = json.loads(line)
                raw_text = example["text"]
                label = example["label"]
                id_ = str(idx)

                # handle special case with square brackets surrounding entities in raw text
                raw_text = re.sub(r"\[\[\[", "[ [[", raw_text)
                raw_text = re.sub(r"\]\]\]", "]] ]", raw_text)
                # handle unicode remnants
                raw_text = re.sub(r"(\u2002|\xa0)", " ", raw_text)

                # TODO check whether adding whitespace before and after symbols may be too aggressive
                raw_text = re.sub(r"([.,!?()])(\S)", r"\1 \2", raw_text)
                raw_text = re.sub(r"(\S)([.,!?()])", r"\1 \2", raw_text)

                # add whitespace before start marker and after end marker
                raw_text = re.sub(r"(\S)(\[\[)", r"\1 \2", raw_text)
                raw_text = re.sub(r"(\S)(<<)", r"\1 \2", raw_text)
                raw_text = re.sub(r"(\]\])(\S)", r"\1 \2", raw_text)
                raw_text = re.sub(r"(>>)(\S)", r"\1 \2", raw_text)

                tokens = raw_text.split(" ")

                assert any(e in tokens for e in ["[[", "]]", "<<", ">>"]), (
                    f"Missing head/tail markers in " f"{example}\n Tokens: {tokens}"
                )

                # Get head/tail order before determining head/tail indices and popping markers
                head_start = tokens.index("[[")
                tail_start = tokens.index("<<")
                if head_start < tail_start:
                    tokens.pop(head_start)
                    head_end = tokens.index("]]")
                    tokens.pop(head_end)
                    tail_start = tokens.index("<<")
                    tokens.pop(tail_start)
                    tail_end = tokens.index(">>")
                    tokens.pop(tail_end)
                else:
                    tokens.pop(tail_start)
                    tail_end = tokens.index(">>")
                    tokens.pop(tail_end)
                    head_start = tokens.index("[[")
                    tokens.pop(head_start)
                    head_end = tokens.index("]]")
                    tokens.pop(head_end)

                yield id_, {
                    "tokens": tokens,
                    "head_start": head_start,
                    "head_end": head_end,
                    "tail_start": tail_start,
                    "tail_end": tail_end,
                    "label": label,
                }
