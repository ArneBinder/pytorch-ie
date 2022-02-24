"""TODO: Add a description here."""


import json

import datasets

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)

_CITATION_FEWREL_1 = """\
@inproceedings{han-etal-2018-fewrel,
    title = "{F}ew{R}el: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation",
    author = "Han, Xu and Zhu, Hao and Yu, Pengfei and Wang, Ziyun and Yao, Yuan and Liu, Zhiyuan and Sun, Maosong",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1514",
    doi = "10.18653/v1/D18-1514",
    pages = "4803--4809"
}"""

_CITATION_FEWREL_2 = """\
@inproceedings{han-etal-2018-fewrel,
    title = "{F}ew{R}el: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation",
    author = "Han, Xu and Zhu, Hao and Yu, Pengfei and Wang, Ziyun and Yao, Yuan and Liu, Zhiyuan and Sun, Maosong",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1514",
    doi = "10.18653/v1/D18-1514",
    pages = "4803--4809"
}

@inproceedings{gao-etal-2019-fewrel,
    title = "{F}ew{R}el 2.0: Towards More Challenging Few-Shot Relation Classification",
    author = "Gao, Tianyu and Han, Xu and Zhu, Hao and Liu, Zhiyuan and Li, Peng and Sun, Maosong and Zhou, Jie",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1649",
    doi = "10.18653/v1/D19-1649",
    pages = "6251--6256"
}
"""


class FewRelConfig(datasets.BuilderConfig):
    """BuilderConfig for FewRel."""

    def __init__(
        self,
        data_url,
        citation,
        url,
        class_labels,
        description,
        **kwargs,
    ):
        """BuilderConfig for FewRel.
        Args:
          data_url: `string`, url to download the zip file from
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          class_labels: `list[string]`, the list of classes if the label is
            categorical. If not provided, then the label will be of type
            `datasets.Value('float32')`.
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.class_labels = class_labels
        self.data_url = data_url
        self.citation = citation
        self.url = url
        self.description = description


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class FewRel(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    BUILDER_CONFIGS = [
        FewRelConfig(
            name="fewrel_train",
            data_url="https://github.com/thunlp/FewRel/raw/master/data/train_wiki.json",
            citation=_CITATION_FEWREL_1,
            url="https://thunlp.github.io/1/fewrel1.html",
            class_labels=[
                "P931",
                "P4552",
                "P140",
                "P1923",
                "P150",
                "P6",
                "P27",
                "P449",
                "P1435",
                "P175",
                "P1344",
                "P39",
                "P527",
                "P740",
                "P706",
                "P84",
                "P495",
                "P123",
                "P57",
                "P22",
                "P178",
                "P241",
                "P403",
                "P1411",
                "P135",
                "P991",
                "P156",
                "P176",
                "P31",
                "P1877",
                "P102",
                "P1408",
                "P159",
                "P3373",
                "P1303",
                "P17",
                "P106",
                "P551",
                "P937",
                "P355",
                "P710",
                "P137",
                "P674",
                "P466",
                "P136",
                "P306",
                "P127",
                "P400",
                "P974",
                "P1346",
                "P460",
                "P86",
                "P118",
                "P264",
                "P750",
                "P58",
                "P3450",
                "P105",
                "P276",
                "P101",
                "P407",
                "P1001",
                "P800",
                "P131",
            ],
            description="",
        ),
        FewRelConfig(
            name="fewrel_validation",
            data_url="https://github.com/thunlp/FewRel/raw/master/data/val_wiki.json",
            citation=_CITATION_FEWREL_1,
            url="https://thunlp.github.io/1/fewrel1.html",
            class_labels=[
                "P177",
                "P364",
                "P2094",
                "P361",
                "P641",
                "P59",
                "P413",
                "P206",
                "P412",
                "P155",
                "P26",
                "P410",
                "P25",
                "P463",
                "P40",
                "P921",
            ],
            description="",
        ),
        FewRelConfig(
            name="fewrel2_validation",
            data_url="https://github.com/thunlp/FewRel/raw/master/data/val_pubmed.json",
            citation=_CITATION_FEWREL_2,
            url="https://thunlp.github.io/2/fewrel2_da.html",
            class_labels=[
                "biological_process_involves_gene_product",
                "inheritance_type_of",
                "is_normal_tissue_origin_of_disease",
                "ingredient_of",
                "is_primary_anatomic_site_of_disease",
                "gene_found_in_organism",
                "occurs_in",
                "causative_agent_of",
                "classified_as",
                "gene_plays_role_in_process",
            ],
            description="",
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "tokens": datasets.Sequence(datasets.Value("string")),
                "head_start": datasets.Value("int32"),
                "head_end": datasets.Value("int32"),
                "tail_start": datasets.Value("int32"),
                "tail_end": datasets.Value("int32"),
                "label": datasets.ClassLabel(names=self.config.class_labels),
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=self.config.description,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=self.config.url,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=self.config.citation,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        filepath = dl_manager.download_and_extract(self.config.data_url)

        split = (
            datasets.Split.VALIDATION if "validation" in self.config.name else datasets.Split.TRAIN
        )

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepath": filepath},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for label, examples in data.items():
                for idx, example in enumerate(examples):
                    id_ = label + "_" + str(idx)

                    head_token_positions = example["h"][2][0]
                    tail_token_positions = example["t"][2][0]

                    head_start = head_token_positions[0]
                    head_end = head_token_positions[-1]
                    tail_start = tail_token_positions[0]
                    tail_end = tail_token_positions[-1]

                    yield id_, {
                        "tokens": example["tokens"],
                        "head_start": head_start,
                        "head_end": head_end + 1,  # make end offset exclusive
                        "tail_start": tail_start,
                        "tail_end": tail_end + 1,  # make end offset exclusive
                        "label": label,
                    }
