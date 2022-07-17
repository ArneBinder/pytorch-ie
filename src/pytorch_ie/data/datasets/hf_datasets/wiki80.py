"""TODO: Add a description here."""


import json

import datasets

_CITATION_WIKI80 = """\
@inproceedings{han-etal-2019-opennre,
    title = "{O}pen{NRE}: An Open and Extensible Toolkit for Neural Relation Extraction",
    author = "Han, Xu and Gao, Tianyu and Yao, Yuan and Ye, Deming and Liu, Zhiyuan and Sun, Maosong",
    booktitle = "Proceedings of EMNLP-IJCNLP: System Demonstrations",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-3029",
    doi = "10.18653/v1/D19-3029",
    pages = "169--174"
}"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
Wiki80 is derived from FewRel, a large
scale few-shot dataset. It contains 80 relations and
56,000 instances from Wikipedia and Wikidata."""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_DATA_URLs = {
    "train": "https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/wiki80/wiki80_train.txt",
    "validation": "https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/wiki80/wiki80_val.txt",
}

_CLASS_LABELS = [
    "place served by transport hub",
    "mountain range",
    "religion",
    "participating team",
    "contains administrative territorial entity",
    "head of government",
    "country of citizenship",
    "original network",
    "heritage designation",
    "performer",
    "participant of",
    "position held",
    "has part",
    "location of formation",
    "located on terrain feature",
    "architect",
    "country of origin",
    "publisher",
    "director",
    "father",
    "developer",
    "military branch",
    "mouth of the watercourse",
    "nominated for",
    "movement",
    "successful candidate",
    "followed by",
    "manufacturer",
    "instance of",
    "after a work by",
    "member of political party",
    "licensed to broadcast to",
    "headquarters location",
    "sibling",
    "instrument",
    "country",
    "occupation",
    "residence",
    "work location",
    "subsidiary",
    "participant",
    "operator",
    "characters",
    "occupant",
    "genre",
    "operating system",
    "owned by",
    "platform",
    "tributary",
    "winner",
    "said to be the same as",
    "composer",
    "league",
    "record label",
    "distributor",
    "screenwriter",
    "sports season of league or competition",
    "taxon rank",
    "location",
    "field of work",
    "language of work or name",
    "applies to jurisdiction",
    "notable work",
    "located in the administrative territorial entity",
    "crosses",
    "original language of film or TV show",
    "competition class",
    "part of",
    "sport",
    "constellation",
    "position played on team / speciality",
    "located in or next to body of water",
    "voice type",
    "follows",
    "spouse",
    "military rank",
    "mother",
    "member of",
    "child",
    "main subject",
]


class Wiki80(datasets.GeneratorBasedBuilder):
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
            citation=_CITATION_WIKI80,
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
                gen_kwargs={"filepath": data_files.get("validation")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                example = json.loads(line)
                label = example["relation"]
                id_ = str(idx)

                head_token_positions = example["h"]["pos"]
                tail_token_positions = example["t"]["pos"]

                head_start = head_token_positions[0]
                head_end = head_token_positions[-1]
                tail_start = tail_token_positions[0]
                tail_end = tail_token_positions[-1]

                yield id_, {
                    "tokens": example["token"],
                    "head_start": head_start,
                    "head_end": head_end,
                    "tail_start": tail_start,
                    "tail_end": tail_end,
                    "label": label,
                }
