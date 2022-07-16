"""TODO: Add a description here."""


import json
import os

import datasets

_CITATION = """\
@inproceedings{zhang-etal-2017-position,
    title = "Position-aware Attention and Supervised Data Improve Slot Filling",
    author = "Zhang, Yuhao  and
      Zhong, Victor  and
      Chen, Danqi  and
      Angeli, Gabor  and
      Manning, Christopher D.",
    booktitle = "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D17-1004",
    doi = "10.18653/v1/D17-1004",
    pages = "35--45",
}

@inproceedings{alt-etal-2020-tacred,
    title = "{TACRED} Revisited: A Thorough Evaluation of the {TACRED} Relation Extraction Task",
    author = "Alt, Christoph  and
      Gabryszak, Aleksandra  and
      Hennig, Leonhard",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.142",
    doi = "10.18653/v1/2020.acl-main.142",
    pages = "1558--1569",
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_PATCH_URLs = {
    "dev": "https://raw.githubusercontent.com/DFKI-NLP/tacrev/master/patch/dev_patch.json",
    "test": "https://raw.githubusercontent.com/DFKI-NLP/tacrev/master/patch/test_patch.json",
}

_CLASS_LABELS = [
    "no_relation",
    "org:alternate_names",
    "org:city_of_headquarters",
    "org:country_of_headquarters",
    "org:dissolved",
    "org:founded",
    "org:founded_by",
    "org:member_of",
    "org:members",
    "org:number_of_employees/members",
    "org:parents",
    "org:political/religious_affiliation",
    "org:shareholders",
    "org:stateorprovince_of_headquarters",
    "org:subsidiaries",
    "org:top_members/employees",
    "org:website",
    "per:age",
    "per:alternate_names",
    "per:cause_of_death",
    "per:charges",
    "per:children",
    "per:cities_of_residence",
    "per:city_of_birth",
    "per:city_of_death",
    "per:countries_of_residence",
    "per:country_of_birth",
    "per:country_of_death",
    "per:date_of_birth",
    "per:date_of_death",
    "per:employee_of",
    "per:origin",
    "per:other_family",
    "per:parents",
    "per:religion",
    "per:schools_attended",
    "per:siblings",
    "per:spouse",
    "per:stateorprovince_of_birth",
    "per:stateorprovince_of_death",
    "per:stateorprovinces_of_residence",
    "per:title",
]


def convert_ptb_token(token: str) -> str:
    """Convert PTB tokens to normal tokens"""
    return {
        "-lrb-": "(",
        "-rrb-": ")",
        "-lsb-": "[",
        "-rsb-": "]",
        "-lcb-": "{",
        "-rcb-": "}",
    }.get(token.lower(), token)


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class TACRED(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="original", version=datasets.Version("1.0.0"), description="The original TACRED."
        ),
        datasets.BuilderConfig(
            name="revised",
            version=datasets.Version("1.0.0"),
            description="The revised TACRED (corrected labels in dev and test split).",
        ),
    ]

    DEFAULT_CONFIG_NAME = "original"  # type: ignore

    @property
    def manual_download_instructions(self):
        return (
            "To use TACRED you have to download it manually. "
            "It is available via the LDC at https://catalog.ldc.upenn.edu/LDC2018T24"
            "Please extract all files in one folder and load the dataset with: "
            "`datasets.load_dataset('tacred', data_dir='path/to/folder/folder_name')`"
        )

    def _info(self):
        features = datasets.Features(
            {
                "tokens": datasets.Sequence(datasets.Value("string")),
                "head_start": datasets.Value("int32"),
                "head_end": datasets.Value("int32"),
                "tail_start": datasets.Value("int32"),
                "tail_end": datasets.Value("int32"),
                "label": datasets.ClassLabel(names=_CLASS_LABELS),
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        patch_files = {}
        if self.config.name == "revised":
            patch_files = dl_manager.download_and_extract(_PATCH_URLs)

        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                "{} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('tacred', data_dir=...)` that includes the unzipped files from the TACRED_LDC zip. Manual download instructions: {}".format(
                    data_dir, self.manual_download_instructions
                )
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.json"),
                    "patch_filepath": None,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.json"),
                    "patch_filepath": patch_files.get("test"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.json"),
                    "patch_filepath": patch_files.get("dev"),
                },
            ),
        ]

    def _generate_examples(self, filepath, patch_filepath):
        """Yields examples."""
        # TODO: This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)
        patch_examples = {}
        if patch_filepath is not None:
            with open(patch_filepath, encoding="utf-8") as f:
                patch_examples = {example["id"]: example for example in json.load(f)}

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data:
                id_ = example["id"]

                if id_ in patch_examples:
                    example.update(patch_examples[id_])

                yield id_, {
                    "tokens": [convert_ptb_token(token) for token in example["token"]],
                    "head_start": example["subj_start"],
                    "head_end": example["subj_end"] + 1,  # make end offset exclusive
                    "tail_start": example["obj_start"],
                    "tail_end": example["obj_end"] + 1,  # make end offset exclusive
                    "label": example["relation"],
                }
