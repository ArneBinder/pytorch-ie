"""The SemEval-2010 Task 8 on Multi-way classification of semantic relations between pairs of nominals"""


import os
import re

import datasets

_CITATION = """\
@inproceedings{hendrickx-etal-2010-semeval,
    title = "{S}em{E}val-2010 Task 8: Multi-Way Classification of Semantic Relations between Pairs of Nominals",
    author = "Hendrickx, Iris  and
      Kim, Su Nam  and
      Kozareva, Zornitsa  and
      Nakov, Preslav  and
      {\'O} S{\'e}aghdha, Diarmuid  and
      Pad{\'o}, Sebastian  and
      Pennacchiotti, Marco  and
      Romano, Lorenza  and
      Szpakowicz, Stan",
    booktitle = "Proceedings of the 5th International Workshop on Semantic Evaluation",
    month = jul,
    year = "2010",
    address = "Uppsala, Sweden",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/S10-1006",
    pages = "33--38",
}
"""

_DESCRIPTION = """\
The SemEval-2010 Task 8 focuses on Multi-way classification of semantic relations between pairs of nominals.
The task was designed to compare different approaches to semantic relation classification
and to provide a standard testbed for future research.
"""

_URL = "https://drive.google.com/uc?export=download&id=0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk"

_CLASS_LABELS = [
    "Cause-Effect(e1,e2)",
    "Cause-Effect(e2,e1)",
    "Component-Whole(e1,e2)",
    "Component-Whole(e2,e1)",
    "Content-Container(e1,e2)",
    "Content-Container(e2,e1)",
    "Entity-Destination(e1,e2)",
    "Entity-Destination(e2,e1)",
    "Entity-Origin(e1,e2)",
    "Entity-Origin(e2,e1)",
    "Instrument-Agency(e1,e2)",
    "Instrument-Agency(e2,e1)",
    "Member-Collection(e1,e2)",
    "Member-Collection(e2,e1)",
    "Message-Topic(e1,e2)",
    "Message-Topic(e2,e1)",
    "Product-Producer(e1,e2)",
    "Product-Producer(e2,e1)",
    "Other",
]


class SemEval2010Task8(datasets.GeneratorBasedBuilder):
    """The SemEval-2010 Task 8 focuses on Multi-way classification of semantic relations between pairs of nominals.
    The task was designed to compare different approaches to semantic relation classification
    and to provide a standard testbed for future research."""

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
            homepage="https://semeval2.fbk.eu/semeval2.php?location=tasks&taskid=11",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "SemEval2010_task8_all_data")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "SemEval2010_task8_training/TRAIN_FILE.TXT"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            raw_lines = []
            for line in f:
                line = line.strip()

                if not line:
                    idx, example = self._raw_lines_to_example(raw_lines)
                    yield idx, example
                    raw_lines = []
                    continue

                raw_lines.append(line)

    def _raw_lines_to_example(self, raw_lines):
        raw_id, raw_text = raw_lines[0].split("\t")
        label = raw_lines[1]
        id_ = int(raw_id)
        raw_text = raw_text.strip('"')

        # Some special cases (e.g., missing spaces before entity marker)
        if id_ in [213, 4612, 6373, 8411, 9867]:
            raw_text = raw_text.replace("<e2>", " <e2>")
        if id_ in [2740, 4219, 4784]:
            raw_text = raw_text.replace("<e1>", " <e1>")
        if id_ == 9256:
            raw_text = raw_text.replace("log- jam", "log-jam")

        # necessary if text should be whitespace tokenizeable
        if id_ in [2609, 7589]:
            raw_text = raw_text.replace("1 1/2", "1-1/2")
        if id_ == 10591:
            raw_text = raw_text.replace("1 1/4", "1-1/4")
        if id_ == 10665:
            raw_text = raw_text.replace("6 1/2", "6-1/2")

        raw_text = re.sub(r"([.,!?()])$", r" \1", raw_text)
        raw_text = re.sub(r"(e[12]>)([',;:\"\(\)])", r"\1 \2", raw_text)
        raw_text = re.sub(r"([',;:\"\(\)])(</?e[12])", r"\1 \2", raw_text)
        raw_text = raw_text.replace("<e1>", "<e1> ")
        raw_text = raw_text.replace("<e2>", "<e2> ")
        raw_text = raw_text.replace("</e1>", " </e1>")
        raw_text = raw_text.replace("</e2>", " </e2>")

        tokens = raw_text.split(" ")

        head_start = tokens.index("<e1>")
        tokens.pop(head_start)

        head_end = tokens.index("</e1>")
        tokens.pop(head_end)

        tail_start = tokens.index("<e2>")
        tokens.pop(tail_start)

        tail_end = tokens.index("</e2>")
        tokens.pop(tail_end)

        return id_, {
            "tokens": tokens,
            "head_start": head_start,
            "head_end": head_end,
            "tail_start": tail_start,
            "tail_end": tail_end,
            "label": label,
        }
