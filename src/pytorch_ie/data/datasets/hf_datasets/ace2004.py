"""TODO: Add a description here."""


import json
import os

import datasets

_CITATION_ACE2004 = """\
@inproceedings{doddington-etal-2004-automatic,
    title = "The Automatic Content Extraction ({ACE}) Program {--} Tasks, Data, and Evaluation",
    author = "Doddington, George  and
      Mitchell, Alexis  and
      Przybocki, Mark  and
      Ramshaw, Lance  and
      Strassel, Stephanie  and
      Weischedel, Ralph",
    booktitle = "Proceedings of the Fourth International Conference on Language Resources and Evaluation ({LREC}{'}04)",
    month = may,
    year = "2004",
    address = "Lisbon, Portugal",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2004/pdf/5.pdf",
}
"""

# You can copy an official description
_DESCRIPTION = """\
ACE 2004 Multilingual Training Corpus contains the complete set of English, Arabic and Chinese
training data for the 2004 Automatic Content Extraction (ACE) technology evaluation. The corpus consists of data of
various types annotated for entities and relations and was created by Linguistic Data Consortium with support from
the ACE Program, with additional assistance from the DARPA TIDES (Translingual Information Detection, Extraction and
Summarization) Program. This data was previously distributed as an e-corpus (LDC2004E17) to participants in the 2004
ACE evaluation.

The objective of the ACE program is to develop automatic content extraction technology to support automatic
processing of human language in text form. In September 2004, sites were evaluated on system performance in six
areas: Entity Detection and Recognition (EDR), Entity Mention Detection (EMD), EDR Co-reference, Relation Detection
and Recognition (RDR), Relation Mention Detection (RMD), and RDR given reference entities. All tasks were evaluated
in three languages: English, Chinese and Arabic.

The current publication consists of the official training data for these evaluation tasks. A seventh evaluation area,
Timex Detection and Recognition, is supported by the ACE Time Normalization (TERN) 2004 English Training Data Corpus
(LDC2005T07). The TERN corpus source data largely overlaps with the English source data contained in the current
release.

For more information about linguistic resources for the ACE program, including annotation guidelines,
task definitions, free annotation tools and other documentation, please visit LDC's ACE website:
https://www.ldc.upenn.edu/collaborations/past-projects/ace
"""

_HOMEPAGE = "https://catalog.ldc.upenn.edu/LDC2005T09"

# TODO: Add the license for the dataset here if you can find it
_LICENSE = """https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf"""

# TODO: Add class labels
_CLASS_LABELS = ["PHYS", "EMP-ORG", "ART", "OTHER-AFF", "GPE-AFF", "PER-SOC"]


class ACE2004(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")  # type: ignore

    @property
    def manual_download_instructions(self):
        return (
            "To use ACE2004 you have to download it manually. "
            "It is available via the LDC at https://catalog.ldc.upenn.edu/LDC2005T09"
            "Preprocess the data as described in "
            "https://github.com/LorrinWWW/two-are-better-than-one/tree/master/datasets and "
            "extract test.ACE04_0,json, train.ACE04_0.json, valid.ACE04_0.json files from the "
            "unified folder in one folder, and load the dataset with: "
            "`datasets.load_dataset('ace2004', data_dir='path/to/folder/folder_name')`"
        )

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
            citation=_CITATION_ACE2004,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                "{} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('ace2004', data_dir=...)` that includes the train, valid, test files. Manual download instructions: {}".format(
                    data_dir, self.manual_download_instructions
                )
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "train.ACE04_0.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "test.ACE04_0.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "valid.ACE04_0.json")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data:
                idx = 0
                for rel in example["relations"]:
                    head_start, head_end, tail_start, tail_end, label = rel

                    id_ = str(idx)
                    idx += 1

                    yield id_, {
                        "tokens": example["tokens"],
                        "head_start": head_start,
                        "head_end": head_end,
                        "tail_start": tail_start,
                        "tail_end": tail_end,
                        "label": label,
                    }
