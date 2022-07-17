"""TODO: Add a description here."""


import json
import os

import datasets

_CITATION_ACE2005 = """\
@article{walker2006ace,
  title={ACE 2005 multilingual training corpus},
  author={Walker, Christopher and Strassel, Stephanie and Medero, Julie and Maeda, Kazuaki},
  journal={Linguistic Data Consortium, Philadelphia},
  volume={57},
  pages={45},
  year={2006}
}
"""

# You can copy an official description
_DESCRIPTION = """\
ACE 2005 Multilingual Training Corpus contains the complete set of English, Arabic and Chinese
training data for the 2005 Automatic Content Extraction (ACE) technology evaluation. The corpus consists of data of
various types annotated for entities, relations and events by the Linguistic Data Consortium (LDC) with support from
the ACE Program and additional assistance from LDC.

The objective of the ACE program was to develop automatic content extraction technology to support automatic
processing of human language in text form.

In November 2005, sites were evaluated on system performance in five primary areas: the recognition of entities,
values, temporal expressions, relations, and events. Entity, relation and event mention detection were also offered
as diagnostic tasks. All tasks with the exception of event tasks were performed for three languages, English,
Chinese and Arabic. Events tasks were evaluated in English and Chinese only. This release comprises the official
training data for these evaluation tasks.

For more information about linguistic resources for the ACE Program, including annotation guidelines,
task definitions and other documentation, see LDC's ACE website:
http://projects.ldc.upenn.edu/ace/
"""

_HOMEPAGE = "https://catalog.ldc.upenn.edu/LDC2006T06"

_LICENSE = """https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf"""

_CLASS_LABELS = ["PHYS", "ART", "PART-WHOLE", "ORG-AFF", "GEN-AFF", "PER-SOC"]


class ACE2004(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")  # type: ignore

    @property
    def manual_download_instructions(self):
        return (
            "To use ACE2005 you have to download it manually. "
            "It is available via the LDC at https://catalog.ldc.upenn.edu/LDC2006T06"
            "Preprocess the data as described in "
            "https://github.com/LorrinWWW/two-are-better-than-one/tree/master/datasets and "
            "extract test.ACE05.json, train.ACE05.json, valid.ACE05.json files from the "
            "unified folder in one folder, and load the dataset with: "
            "`datasets.load_dataset('ace2005', data_dir='path/to/folder/folder_name')`"
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
            citation=_CITATION_ACE2005,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                "{} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('ace2005', data_dir=...)` that includes the train, valid, test files. Manual download instructions: {}".format(
                    data_dir, self.manual_download_instructions
                )
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "train.ACE05.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "test.ACE05.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "valid.ACE05.json")},
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
