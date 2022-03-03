import glob
import logging
from dataclasses import dataclass
from os import listdir, path
from typing import Dict, List, Optional

import datasets
from datasets import BuilderConfig, DatasetInfo, Features, Sequence, SplitGenerator, Value

logger = logging.getLogger(__name__)


@dataclass
class BratConfig(BuilderConfig):
    """BuilderConfig for BRAT."""

    url: str = None  # type: ignore
    description: Optional[str] = None
    citation: Optional[str] = None
    homepage: Optional[str] = None

    subdirectory_mapping: Optional[Dict[str, str]] = None
    file_name_blacklist: Optional[List[str]] = None
    ann_file_extension: str = "ann"
    txt_file_extension: str = "txt"


class Brat(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = BratConfig

    def _info(self):
        return DatasetInfo(
            description=self.config.description,
            citation=self.config.citation,
            homepage=self.config.homepage,
            features=Features(
                {
                    "context": Value("string"),
                    "file_name": Value("string"),
                    "spans": Sequence(
                        {
                            "id": Value("string"),
                            "type": Value("string"),
                            "locations": Sequence(
                                {
                                    "start": Value("int32"),
                                    "end": Value("int32"),
                                }
                            ),
                            "text": Value("string"),
                        }
                    ),
                    "relations": Sequence(
                        {
                            "id": Value("string"),
                            "type": Value("string"),
                            "arguments": Sequence(
                                {"type": Value("string"), "target": Value("string")}
                            ),
                        }
                    ),
                    "equivalence_relations": Sequence(
                        {
                            "type": Value("string"),
                            "targets": Sequence(Value("string")),
                        }
                    ),
                    "events": Sequence(
                        {
                            "id": Value("string"),
                            "type": Value("string"),
                            "trigger": Value("string"),
                            "arguments": Sequence(
                                {"type": Value("string"), "target": Value("string")}
                            ),
                        }
                    ),
                    "attributions": Sequence(
                        {
                            "id": Value("string"),
                            "type": Value("string"),
                            "target": Value("string"),
                            "value": Value("string"),
                        }
                    ),
                    "normalizations": Sequence(
                        {
                            "id": Value("string"),
                            "type": Value("string"),
                            "target": Value("string"),
                            "resource_id": Value("string"),
                            "entity_id": Value("string"),
                        }
                    ),
                    "notes": Sequence(
                        {
                            "id": Value("string"),
                            "type": Value("string"),
                            "target": Value("string"),
                            "note": Value("string"),
                        }
                    ),
                }
            ),
        )

    @staticmethod
    def _get_location(location_string):
        parts = location_string.split(" ")
        assert (
            len(parts) == 2
        ), f"Wrong number of entries in location string. Expected 2, but found: {parts}"
        return {"start": int(parts[0]), "end": int(parts[1])}

    @staticmethod
    def _get_span_annotation(annotation_line):
        """
        example input:
        T1	Organization 0 4	Sony
        """

        _id, remaining, text = annotation_line.split("\t", maxsplit=2)
        _type, locations = remaining.split(" ", maxsplit=1)
        return {
            "id": _id,
            "text": text,
            "type": _type,
            "locations": [Brat._get_location(loc) for loc in locations.split(";")],
        }

    @staticmethod
    def _get_event_annotation(annotation_line):
        """
        example input:
        E1	MERGE-ORG:T2 Org1:T1 Org2:T3
        """
        _id, remaining = annotation_line.strip().split("\t")
        args = [dict(zip(["type", "target"], a.split(":"))) for a in remaining.split(" ")]
        return {
            "id": _id,
            "type": args[0]["type"],
            "trigger": args[0]["target"],
            "arguments": args[1:],
        }

    @staticmethod
    def _get_relation_annotation(annotation_line):
        """
        example input:
        R1	Origin Arg1:T3 Arg2:T4
        """

        _id, remaining = annotation_line.strip().split("\t")
        _type, remaining = remaining.split(" ", maxsplit=1)
        args = [dict(zip(["type", "target"], a.split(":"))) for a in remaining.split(" ")]
        return {"id": _id, "type": _type, "arguments": args}

    @staticmethod
    def _get_equivalence_relation_annotation(annotation_line):
        """
        example input:
        *	Equiv T1 T2 T3
        """
        _, remaining = annotation_line.strip().split("\t")
        parts = remaining.split(" ")
        return {"type": parts[0], "targets": parts[1:]}

    @staticmethod
    def _get_attribute_annotation(annotation_line):
        """
        example input (binary: implicit value is True, if present, False otherwise):
        A1	Negation E1
        example input (multi-value: explicit value)
        A2	Confidence E2 L1
        """

        _id, remaining = annotation_line.strip().split("\t")
        parts = remaining.split(" ")
        # if no value is present, it is implicitly "true"
        if len(parts) == 2:
            parts.append("true")
        return {
            "id": _id,
            "type": parts[0],
            "target": parts[1],
            "value": parts[2],
        }

    @staticmethod
    def _get_normalization_annotation(annotation_line):
        """
        example input:
        N1	Reference T1 Wikipedia:534366	Barack Obama
        """
        _id, remaining, text = annotation_line.split("\t", maxsplit=2)
        _type, target, ref = remaining.split(" ")
        res_id, ent_id = ref.split(":")
        return {
            "id": _id,
            "type": _type,
            "target": target,
            "resource_id": res_id,
            "entity_id": ent_id,
        }

    @staticmethod
    def _get_note_annotation(annotation_line):
        """
        example input:
        #1	AnnotatorNotes T1	this annotation is suspect
        """
        _id, remaining, note = annotation_line.split("\t", maxsplit=2)
        _type, target = remaining.split(" ")
        return {
            "id": _id,
            "type": _type,
            "target": target,
            "note": note,
        }

    @staticmethod
    def _read_annotation_file(filename):
        """
        reads a BRAT v1.3 annotations file (see https://brat.nlplab.org/standoff.html)
        """

        res = {
            "spans": [],
            "events": [],
            "relations": [],
            "equivalence_relations": [],
            "attributions": [],
            "normalizations": [],
            "notes": [],
        }

        with open(filename) as file:
            for i, line in enumerate(file):
                if len(line.strip()) == 0:
                    continue
                ann_type = line[0]

                # strip away the new line character
                if line.endswith("\n"):
                    line = line[:-1]

                if ann_type == "T":
                    res["spans"].append(Brat._get_span_annotation(line))
                elif ann_type == "E":
                    res["events"].append(Brat._get_event_annotation(line))
                elif ann_type == "R":
                    res["relations"].append(Brat._get_relation_annotation(line))
                elif ann_type == "*":
                    res["equivalence_relations"].append(
                        Brat._get_equivalence_relation_annotation(line)
                    )
                elif ann_type in ["A", "M"]:
                    res["attributions"].append(Brat._get_attribute_annotation(line))
                elif ann_type == "N":
                    res["normalizations"].append(Brat._get_normalization_annotation(line))
                elif ann_type == "#":
                    res["notes"].append(Brat._get_note_annotation(line))
                else:
                    raise ValueError(
                        f'unknown BRAT annotation id type: "{line}" (from file {filename} @line {i}). '
                        f"Annotation ids have to start with T (spans), E (events), R (relations), "
                        f"A (attributions), or N (normalizations). See "
                        f"https://brat.nlplab.org/standoff.html for the BRAT annotation file "
                        f"specification."
                    )
        return res

    def _generate_examples(self, files=None, directory=None):
        """Read context (.txt) and annotation (.ann) files."""
        if files is None:
            assert (
                directory is not None
            ), "If files is None, directory has to be provided, but it is also None."
            _files = glob.glob(f"{directory}/*.{self.config.ann_file_extension}")
            files = sorted(path.splitext(fn)[0] for fn in _files)

        for filename in files:
            basename = path.basename(filename)
            if (
                self.config.file_name_blacklist is not None
                and basename in self.config.file_name_blacklist
            ):
                logger.info(f"skip annotation file: {basename} (blacklisted)")
                continue

            ann_fn = f"{filename}.{self.config.ann_file_extension}"
            brat_annotations = Brat._read_annotation_file(ann_fn)

            txt_fn = f"{filename}.{self.config.txt_file_extension}"
            txt_content = open(txt_fn).read()
            brat_annotations["context"] = txt_content
            brat_annotations["file_name"] = basename

            yield basename, brat_annotations

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        subdirectory_mapping = self.config.subdirectory_mapping

        # since subclasses of BuilderConfig are not allowed to define
        # attributes without defaults, check here
        assert self.config.url is not None, "data url not specified"

        # if url points to a local directory, just point to that
        if path.exists(self.config.url) and path.isdir(self.config.url):
            data_dir = self.config.url
        # otherwise, download and extract
        else:
            data_dir = dl_manager.download_and_extract(self.config.url)
        logging.info(f"load from data dir: {data_dir}")

        # if no subdirectory mapping is provided, ...
        if subdirectory_mapping is None:
            # ... use available subdirectories as split names ...
            subdirs = [f for f in listdir(data_dir) if path.isdir(path.join(data_dir, f))]
            if len(subdirs) > 0:
                subdirectory_mapping = {subdir: subdir for subdir in subdirs}
            else:
                # ... otherwise, default to a single train split with the base directory
                subdirectory_mapping = {"": "train"}

        return [
            SplitGenerator(
                name=split,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "directory": path.join(data_dir, subdir),
                },
            )
            for subdir, split in subdirectory_mapping.items()
        ]
