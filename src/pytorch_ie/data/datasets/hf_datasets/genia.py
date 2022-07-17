"""TODO: Add a description here."""


import os

import spacy
from spacy.lang.en import English
from spacy.symbols import ORTH

import datasets

_CITATION_GENIA = """\
@article{article,
    author = {Kim, Jin-Dong and Ohta, Tomoko and Tateisi, Yuka and Tsujii, Jun'ichi},
    year = {2003},
    month = {02},
    pages = {i180-2},
    title = {GENIA corpus—A semantically annotated corpus for bio-textmining},
    volume = {19 Suppl 1},
    journal = {Bioinformatics (Oxford, England)},
    doi = {10.1093/bioinformatics/btg1023}
}"""

# You can copy an official description
_DESCRIPTION = """
The GENIA corpus is the primary collection of biomedical literature compiled and annotated within the scope
of the GENIA project. The corpus was created to support the development and evaluation of information
extraction and text mining systems for the domain of molecular biology.
"""

_HOMEPAGE = "http://www.geniaproject.org/genia-corpus/relation-corpus"

# TODO: Add the license for the dataset here if you can find it
_LICENSE = """\
GENIA Project License for Annotated Corpora

1. Copyright of abstracts

Any abstracts contained in this corpus are from PubMed(R), a database
of the U.S. National Library of Medicine (NLM).

NLM data are produced by a U.S. Government agency and include works of
the United States Government that are not protected by U.S. copyright
law but may be protected by non-US copyright law, as well as abstracts
originating from publications that may be protected by U.S. copyright
law.

NLM assumes no responsibility or liability associated with use of
copyrighted material, including transmitting, reproducing,
redistributing, or making commercial use of the data. NLM does not
provide legal advice regarding copyright, fair use, or other aspects
of intellectual property rights. Persons contemplating any type of
transmission or reproduction of copyrighted material such as abstracts
are advised to consult legal counsel.

2. Copyright of full texts

Any full texts contained in this corpus are from the PMC Open Access
Subset of PubMed Central (PMC), the U.S. National Institutes of Health
(NIH) free digital archive of biomedical and life sciences journal
literature.

Articles in the PMC Open Access Subset are protected by copyright, but
are made available under a Creative Commons or similar license that
generally allows more liberal redistribution and reuse than a
traditional copyrighted work. Please refer to the license of each
article for specific license terms.

3. Copyright of annotations

The copyrights of annotations created in the GENIA Project of Tsujii
Laboratory, University of Tokyo, belong in their entirety to the GENIA
Project.

4. Licence terms

Use and distribution of abstracts drawn from PubMed is subject to the
PubMed(R) license terms as stated in Clause 1.

Use and distribution of full texts is subject to the license terms
applying to each publication.

Annotations created by the GENIA Project are licensed under the
Creative Commons Attribution 3.0 Unported License. To view a copy of
this license, visit http://creativecommons.org/licenses/by/3.0/ or
send a letter to Creative Commons, 444 Castro Street, Suite 900,
Mountain View, California, 94041, USA.

Annotations created by the GENIA Project must be attributed as
detailed in Clause 5.

5. Attribution

The GENIA Project was founded and led by prof. Jun'ichi Tsujii and
the project and its annotation efforts have been coordinated in part
by Nigel Collier, Yuka Tateisi, Sang-Zoo Lee, Tomoko Ohta, Jin-Dong
Kim, and Sampo Pyysalo.

For a complete list of the GENIA Project members and contributors,
please refer to http://www.geniaproject.org.

The GENIA Project has been supported by Grant-in-Aid for Scientific
Research on Priority Area "Genome Information Science" (MEXT, Japan),
Grant-in-Aid for Scientific Research on Priority Area "Systems
Genomics" (MEXT, Japan), Core Research for Evolutional Science &
Technology (CREST) "Information Mobility Project" (JST, Japan),
Solution Oriented Research for Science and Technology (SORST) (JST,
Japan), Genome Network Project (MEXT, Japan) and Grant-in-Aid for
Specially Promoted Research (MEXT, Japan).

Annotations covered by this license must be attributed as follows:

    Corpus annotations (c) GENIA Project

Distributions including annotations covered by this licence must
include this license text and Attribution section.

6. References

- GENIA Project : http://www.geniaproject.org
- PubMed : http://www.pubmed.gov/
- NLM (United States National Library of Medicine) : http://www.nlm.nih.gov/
- MEXT (Ministry of Education, Culture, Sports, Science and Technology) : http://www.mext.go.jp/
- JST (Japan Science and Technology Agency) : http://www.jst.go.jp
"""

# TODO: Add link to the official dataset URLs here, currently test points to blind test file
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_DATA_URLs = {
    "train": "http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Relation/GENIA_relation_annotation_training_data.tar.gz",
    "dev": "http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Relation/GENIA_relation_annotation_development_data.tar.gz",
    # "test": "http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Relation/GENIA_relation_annotation_test_data.tar.gz"
}
# TODO: Add class labels
_CLASS_LABELS = ["Subunit-Complex", "Protein-Component"]


class Genia(datasets.GeneratorBasedBuilder):
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
            citation=_CITATION_GENIA,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
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
                gen_kwargs={"filepath": data_files.get("dev")},
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={"filepath": data_files.get("test")},
            # ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        doc_ids, list_of_files = self._get_doc_ids_and_file_paths(filepath)
        processed_docs = self._get_processed_docs(doc_ids, list_of_files)

        idx = 0
        for doc in processed_docs:
            if "sentences" in doc and "sent_rels" in doc:
                sent_start_index = 0
                for sent, rels in zip(doc["sentences"], doc["sent_rels"]):
                    for rel in rels:
                        label = rel["label"]
                        head_start = rel["head_start"] - sent_start_index
                        head_end = rel["head_end"] - sent_start_index
                        tail_start = rel["tail_start"] - sent_start_index
                        tail_end = rel["tail_end"] - sent_start_index

                        id_ = str(idx) + "_" + doc["doc_id"]
                        idx += 1

                        yield id_, {
                            "tokens": sent["tokens"],
                            "head_start": head_start,
                            "head_end": head_end,
                            "tail_start": tail_start,
                            "tail_end": tail_end,
                            "label": label,
                        }

                    sent_start_index += len(sent)
            else:
                for rel in doc["relations"]:
                    label = rel["label"]
                    head_start = rel["head_start"]
                    head_end = rel["head_end"]
                    tail_start = rel["tail_start"]
                    tail_end = rel["tail_end"]

                    id_ = str(idx) + "_" + doc["doc_id"]
                    idx += 1

                    yield id_, {
                        "tokens": doc["tokens"],
                        "head_start": head_start,
                        "head_end": head_end,
                        "tail_start": tail_start,
                        "tail_end": tail_end,
                        "label": label,
                    }

    def _get_doc_ids_and_file_paths(self, path):
        list_of_files = {}
        for root, dirs, files in os.walk(path):
            for file in files:
                if file not in ["LICENSE", "README"]:
                    list_of_files[file] = os.path.join(root, file)
        doc_ids = list({file_name.split(".")[0] for file_name in list_of_files.keys()})
        doc_ids.sort()
        doc_ids.sort(key=len)
        return doc_ids, list_of_files

    def _get_processed_docs(self, doc_ids, list_of_files):
        ssplit = False
        try:
            nlp = spacy.load("en_core_web_sm")
            special_case = [{ORTH: "ca."}]
            nlp.tokenizer.add_special_case("ca.", special_case)
            ssplit = True
        except OSError as e:
            print(e)
            print(
                "You have to download the model first to enable sentence splitting: "
                "\tpython -m spacy download en_core_web_sm"
            )
            print("Resorting to tokenization only")
            nlp = English()
        processed_docs = []
        for doc_id in doc_ids:
            try:
                txt_file = list_of_files[doc_id + ".txt"]
                a1_file = list_of_files[doc_id + ".a1"]
                rel_file = list_of_files[doc_id + ".rel"]
            except KeyError:
                print(f"Missing annotation file for doc {doc_id}")
                continue

            relations = []
            entities = {}
            with open(txt_file, encoding="utf-8") as txt:
                text = txt.read()
            doc = nlp(text)
            with open(a1_file, encoding="utf-8") as a1:
                for line in a1.readlines():
                    if line.startswith("T"):
                        entity_id, entity = self._retrieve_entity(line, doc, doc_id)
                        entities[entity_id] = entity
            with open(rel_file, encoding="utf-8") as rel:
                for line in rel.readlines():
                    if line.startswith("T"):
                        entity_id, entity = self._retrieve_entity(line, doc, doc_id)
                        entities[entity_id] = entity
                    elif line.startswith("R"):
                        relations.append(self._retrieve_relation(line, entities))
            tokens = [token.text for token in doc]
            processed_doc = {
                "doc_id": doc_id,
                "text": text,
                "tokens": tokens,
                "entities": entities,
                "relations": relations,
            }
            if ssplit:
                sentences = self._convert_sentences(doc.sents)
                sentences = self._fix_ssplit(doc_id, sentences)
                sentence_tokens = []
                sentence_relations = []
                left_over_rels_indices = [True for _ in relations]
                for sent in sentences:
                    sent_rels = []
                    for idx, relation in enumerate(relations):
                        if (
                            min(relation["head_start"], relation["tail_start"]) >= sent["start"]
                            and max(relation["head_end"], relation["tail_end"]) <= sent["end"]
                        ):
                            sent_rels.append(relation)
                            left_over_rels_indices[idx] = False
                    sentence_tokens.append(sent["tokens"])
                    sentence_relations.append(sent_rels)
                left_over_rels = []
                for indicator, relation in zip(left_over_rels_indices, relations):
                    if indicator:
                        left_over_rels.append(relation)
                if left_over_rels:
                    print(
                        f"Examples in doc {doc_id} where spaCy ssplit were not compatible with relation annotation:"
                    )
                    print([list(sent) for sent in doc.sents])
                    print(sentences)
                    print(left_over_rels)
                processed_doc["sentences"] = sentences
                processed_doc["sent_rels"] = sentence_relations
            processed_docs.append(processed_doc)
        return processed_docs

    def _retrieve_entity(self, line, doc, doc_id=""):
        cols = line.strip().split()
        entity_id, _, start_char, end_char = cols[0:4]
        start_char, end_char = int(start_char), int(end_char)
        entity_type = " ".join(cols[4:])
        # default alignment mode is strict, but charOffset in annotation sometimes does not translate to token offsets
        # well, e.g. charOffsets only cover "LMP1" in "LMP1+"
        span = doc.char_span(start_char, end_char, alignment_mode="expand")
        if span:
            start, end = span.start, span.end
        else:
            snippet_start = max(0, start_char - 10)
            snippet_end = min(len(doc.text), end_char + 10)
            raise ValueError(
                f"{doc_id} Could not retrieve span for character offsets: "
                f"text[{start_char},{end_char}] = {doc.text[start_char:end_char]}\n"
                f"{doc.text[snippet_start:snippet_end]}\n"
                f"{list(doc)}"
            )
        return (entity_id, {"start": start, "end": end, "entity_type": entity_type})

    def _retrieve_relation(self, line, entities):
        cols = line.strip().split()
        relation_id, rel_type, arg1, arg2 = cols
        arg1 = arg1.split(":")[-1]
        head_start, head_end = entities[arg1]["start"], entities[arg1]["end"]
        arg2 = arg2.split(":")[-1]
        tail_start, tail_end = entities[arg2]["start"], entities[arg2]["end"]
        return {
            "rel_id": relation_id,
            "head_start": head_start,
            "head_end": head_end,
            "tail_start": tail_start,
            "tail_end": tail_end,
            "label": rel_type,
        }

    def _convert_sentences(self, sentences):
        sentence_dicts = []
        for sent in sentences:
            start, end = sent.start, sent.end
            tokens = [token.text for token in sent]
            sentence_dicts.append({"tokens": tokens, "start": start, "end": end})
        return sentence_dicts

    def _fix_ssplit(self, doc_id, sentences):
        if doc_id == "PMID-8164652":
            sentences[2]["tokens"] += sentences[3]["tokens"]
            sentences[2]["end"] = sentences[3]["end"]
            del sentences[3]
        elif doc_id == "PMID-9442380":
            sentences[4]["tokens"].append(sentences[5]["tokens"].pop(0))
            sentences[4]["end"] += 1
            sentences[5]["start"] += 1
        elif doc_id == "PMID-10201929":
            sentences[4]["tokens"] += sentences[5]["tokens"]
            sentences[4]["end"] = sentences[5]["end"]
            del sentences[5]
        elif doc_id == "PMID-10428853":
            sentences[3]["tokens"] += sentences[4]["tokens"]
            sentences[3]["end"] = sentences[4]["end"]
            del sentences[4]
        elif doc_id == "PMID-1675604":
            sentences[2]["tokens"] += sentences[3]["tokens"]
            sentences[2]["end"] = sentences[3]["end"]
            del sentences[3]
        sentences = [sent for sent in sentences if sent["tokens"]]
        return sentences
