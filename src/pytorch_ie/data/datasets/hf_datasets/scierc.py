"""TODO: Add a description here."""


import json
import os

import datasets

_CITATION_SCIERC = """\
@InProceedings{luan2018multitask,
     author = {Luan, Yi and He, Luheng and Ostendorf, Mari and Hajishirzi, Hannaneh},
     title = {Multi-Task Identification of Entities, Relations, and Coreferencefor Scientific Knowledge Graph Construction},
     booktitle = {Proc.\\ Conf. Empirical Methods Natural Language Process. (EMNLP)},
     year = {2018},
}"""

# You can copy an official description
_DESCRIPTION = """\
SCIERC includes annotations for scientific entities, their relations, and coreference clusters
for 500 scientific abstracts. These abstracts are taken from 12 AI conference/workshop proceedings
in four AI communities, from the Semantic Scholar Corpus. SCI-ERC extends previous datasets in scientific
articles SemEval 2017 Task 10 and SemEval 2018 Task 7 by extending entity types, relation types, relation coverage,
and adding cross-sentence relations using coreference links.
"""

_HOMEPAGE = "http://nlp.cs.washington.edu/sciIE/"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_DATA_URL = "http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"

_CLASS_LABELS = [
    "USED-FOR",
    "FEATURE-OF",
    "HYPONYM-OF",
    "PART-OF",
    "COMPARE",
    "CONJUNCTION",
    "EVALUATE-FOR",  # label in the data is not documented in annotation guidelines
]


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class SCIERC(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")  # type: ignore

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
            citation=_CITATION_SCIERC,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        dl_dir = dl_manager.download_and_extract(_DATA_URL)
        data_dir = os.path.join(dl_dir, "processed_data/json")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "train.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "dev.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "test.json")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            idx = 0
            for line in f.readlines():
                example = json.loads(line)
                sent_start_index = 0
                for sent, rels in zip(example["sentences"], example["relations"]):
                    for rel in rels:
                        head_start, head_end, tail_start, tail_end, label = rel
                        head_start -= sent_start_index
                        head_end -= sent_start_index
                        tail_start -= sent_start_index
                        tail_end -= sent_start_index

                        id_ = str(idx) + "_" + example["doc_key"]
                        idx += 1

                        yield id_, {
                            "tokens": sent,
                            "head_start": head_start,
                            "head_end": head_end + 1,  # make end offset exclusive
                            "tail_start": tail_start,
                            "tail_end": tail_end + 1,  # make end offset exclusive
                            "label": label,
                        }

                    sent_start_index += len(sent)
