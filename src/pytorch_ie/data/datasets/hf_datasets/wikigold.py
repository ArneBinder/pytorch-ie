from tqdm import tqdm

import datasets

_CITATION = """
@inproceedings{balasuriya-etal-2009-named,
    title = "Named Entity Recognition in Wikipedia",
    author = "Balasuriya, Dominic  and
      Ringland, Nicky  and
      Nothman, Joel  and
      Murphy, Tara  and
      Curran, James R.",
    booktitle = "Proceedings of the 2009 Workshop on The People{'}s Web Meets {NLP}:
    Collaboratively Constructed Semantic Resources (People{'}s Web)",
    month = aug,
    year = "2009",
    address = "Suntec, Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W09-3302",
    pages = "10--18",
}
"""

_LICENCE = "CC-BY 4.0"

_DESCRIPTION = """
WikiGold dataset.
"""

_URL = (
    "https://github.com/juand-r/entity-recognition-datasets/raw/master/"
    "data/wikigold/CONLL-format/data/wikigold.conll.txt"
)

# the label ids
NER_TAGS_DICT = {
    "O": 0,
    "PER": 1,
    "LOC": 2,
    "ORG": 3,
    "MISC": 4,
}


class WikiGoldConfig(datasets.BuilderConfig):
    """BuilderConfig for WikiGold"""

    def __init__(self, **kwargs):
        """BuilderConfig for WikiGold.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class WikiGold(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.features.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.features.Sequence(
                        datasets.features.ClassLabel(names=["O", "PER", "LOC", "ORG", "MISC"])
                    ),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            license=_LICENCE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": urls_to_download},
            ),
        ]

    def _generate_examples(self, filepath=None):
        num_lines = sum(1 for _ in open(filepath))
        id = 0

        with open(filepath) as f:
            tokens, ner_tags = [], []
            for line in tqdm(f, total=num_lines):
                line = line.strip().split()

                if line:
                    assert len(line) == 2
                    token, ner_tag = line

                    if token == "-DOCSTART-":
                        continue

                    tokens.append(token)
                    if ner_tag != "O":
                        ner_tag = ner_tag.split("-")[1]
                    ner_tags.append(NER_TAGS_DICT[ner_tag])

                elif tokens:
                    # organize a record to be written into json
                    record = {
                        "tokens": tokens,
                        "id": str(id),
                        "ner_tags": ner_tags,
                    }
                    tokens, ner_tags = [], []
                    id += 1
                    yield record["id"], record

            # take the last sentence
            if tokens:
                record = {
                    "tokens": tokens,
                    "id": str(id),
                    "ner_tags": ner_tags,
                }
                yield record["id"], record
