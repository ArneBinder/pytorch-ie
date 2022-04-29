import os

from tqdm import tqdm

import datasets

_CITATION = """
"""

_DESCRIPTION = """
OntoNotes 5.0
"""

_URL = (
    "https://cloud.dfki.de/owncloud/index.php/s/S8pB4xTBZ3zQEic/download/OntoNotes-5.0-NER-BIO.zip"
)

_LICENCE = "LDC User Agreement for Non-Members"

# the label ids for ner_tags
NER_TAGS_DICT = {
    "O": 0,
    "CARDINAL": 1,
    "DATE": 2,
    "EVENT": 3,
    "FAC": 4,
    "GPE": 5,
    "LANGUAGE": 6,
    "LAW": 7,
    "LOC": 8,
    "MONEY": 9,
    "NORP": 10,
    "ORDINAL": 11,
    "ORG": 12,
    "PERCENT": 13,
    "PERSON": 14,
    "PRODUCT": 15,
    "QUANTITY": 16,
    "TIME": 17,
    "WORK_OF_ART": 18,
}


class OntoNotesConfig(datasets.BuilderConfig):
    """BuilderConfig for OntoNotes"""

    def __init__(self, **kwargs):
        """BuilderConfig for OntoNotes.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class OntoNotes(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.features.Sequence(datasets.Value("string")),
                    "pos_tags": datasets.features.Sequence(datasets.Value("string")),
                    "parsing": datasets.features.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.features.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "CARDINAL",
                                "DATE",
                                "EVENT",
                                "FAC",
                                "GPE",
                                "LANGUAGE",
                                "LAW",
                                "LOC",
                                "MONEY",
                                "NORP",
                                "ORDINAL",
                                "ORG",
                                "PERCENT",
                                "PERSON",
                                "PRODUCT",
                                "QUANTITY",
                                "TIME",
                                "WORK_OF_ART",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://catalog.ldc.upenn.edu/LDC2013T19",
            citation=_CITATION,
            license=_LICENCE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        urls_to_download,
                        "onto.train.ner",
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(urls_to_download, "onto.development.ner")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(urls_to_download, "onto.test.ner")},
            ),
        ]

    def _generate_examples(self, filepath=None):
        num_lines = sum(1 for _ in open(filepath))
        id = 0

        with open(filepath) as f:
            tokens, pos_tags, dependencies, ner_tags = [], [], [], []
            for line in tqdm(f, total=num_lines):
                line = line.strip().split()

                if line:
                    assert len(line) == 4
                    token, pos_tag, dependency, ner_tag = line
                    if ner_tag != "O":
                        ner_tag = ner_tag.split("-")[1]
                    tokens.append(token)
                    pos_tags.append(pos_tag)
                    dependencies.append(dependency)
                    ner_tags.append(NER_TAGS_DICT[ner_tag])

                elif tokens:
                    # organize a record to be written into json
                    record = {
                        "tokens": tokens,
                        "id": str(id),
                        "pos_tags": pos_tags,
                        "parsing": dependencies,
                        "ner_tags": ner_tags,
                    }
                    tokens, pos_tags, dependencies, ner_tags = [], [], [], []
                    id += 1
                    yield record["id"], record

            # take the last sentence
            if tokens:
                record = {
                    "tokens": tokens,
                    "id": str(id),
                    "pos_tags": pos_tags,
                    "parsing": dependencies,
                    "ner_tags": ner_tags,
                }
                yield record["id"], record
