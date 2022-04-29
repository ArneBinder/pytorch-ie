"""TODO: Add a description here."""


import os
import re

import tensorflow as tf
from spacy.lang.en import English

import datasets

_CITATION_WEBRED = """\
@misc{ormandi2021webred,
    title={WebRED: Effective Pretraining And Finetuning For Relation Extraction On The Web},
    author={Robert Ormandi and Mohammad Saleh and Erin Winter and Vinay Rao},
    year={2021},
    eprint={2102.09681},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2102.09681},
}
"""

# You can copy an official description
_DESCRIPTION = """\
A dataset for extracting relationships from a variety of text found on the World Wide Web. Text
on the web has diverse surface forms including writing styles, complexity and grammar. This dataset collects
sentences from a variety of webpages and documents that represent a variety of those categories. In each sentence,
there will be a subject and object entities tagged with subject SUBJ{...} and object OBJ{...}, respectively. The two
entities are either related by a relation from a set of pre-defined ones or has no relation.

More information about the dataset can be found in our paper: https://arxiv.org/abs/2102.09681
"""

_HOMEPAGE = "https://github.com/google-research-datasets/WebRED"

_LICENSE = """\
This data is licensed by Google LLC under a Creative Commons Attribution 4.0 International License (
http://creativecommons.org/licenses/by/4.0/) Users will be allowed to modify and repost it, and we encourage them to
analyze and publish research based on the data.
"""

# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_DATA_URL = "https://github.com/google-research-datasets/WebRED"

_CLASS_LABELS = [
    "based on",
    "part of the series",
    "drug used for treatment",
    "architectural style",
    "writable file format",
    "work location",
    "position held",
    "followed by",
    "flash point",
    "indigenous to",
    "Mohs' hardness",
    "political alignment",
    "located in protected area",
    "translator",
    "director",
    "highest judicial authority",
    "producer",
    "compressive modulus of elasticity",
    "series spin-off",
    "quantity",
    "lyrics by",
    "cell component",
    "medical condition treated",
    "place of death",
    "number of seats",
    "record label",
    "league level above",
    "military branch",
    "origin of the watercourse",
    "diameter",
    "conversion to SI unit",
    "works in collection",
    "presenter",
    "chairperson",
    "temperature",
    "currency",
    "frequency",
    "standards body",
    "manufacturer",
    "location of final assembly",
    "coat of arms",
    "astronaut mission",
    "length",
    "publication date",
    "place of publication",
    "country of citizenship",
    "minimal lethal dose",
    "game mechanics",
    "afflicts",
    "used by",
    "oxidation state",
    "mother",
    "affiliation",
    "head of state",
    "creator",
    "defendant",
    "head coach of sports team",
    "country",
    "developer",
    "approved by",
    "cover artist",
    "lake inflows",
    "separated from",
    "operating area",
    "water as percent of area",
    "head coach",
    "update method",
    "floruit",
    "party chief representative",
    "commander of",
    "gestation period",
    "religious order",
    "school district",
    "depicted by",
    "publisher",
    "excavation director",
    "airline alliance",
    "librettist",
    "executive producer",
    "donated by",
    "mushroom ecological type",
    "iconographic symbol",
    "speed limit",
    "number of representatives in an organization/legislature",
    "subsidiary",
    "educated at",
    "number of participants",
    "founded by",
    "country of origin",
    "family",
    "package management system",
    "subject has role",
    "sibling",
    "interchange station",
    "facet of",
    "decays to",
    "repeals",
    "legislative body",
    "occupant",
    "atomic number",
    "CPU",
    "GUI toolkit or framework",
    "has parts of the class",
    "director of photography",
    "shares border with",
    "parent organization",
    "population",
    "upper flammable limit",
    "performer",
    "isospin z-component",
    "number of injured",
    "number of seasons",
    "choreographer",
    "replaces",
    "doctoral advisor",
    "official residence",
    "top-level Internet domain",
    "VAT-rate",
    "point in time",
    "distance from Earth",
    "public holiday",
    "languages spoken, written or signed",
    "located on astronomical location",
    "solved by",
    "designed by",
    "twinned administrative body",
    "encoded by",
    "located in time zone",
    "canonization status",
    "date of official opening",
    "student",
    "brand",
    "refractive index",
    "inflation rate",
    "home venue",
    "neutron number",
    "chief operating officer",
    "lowest point",
    "signatory",
    "consecrator",
    "model item",
    "time of earliest written record",
    "area",
    "terminus location",
    "significant event",
    "inspired by",
    "backup or reserve team or crew",
    "maximum number of players",
    "talk show guest",
    "number of deaths",
    "exclave of",
    "maximal incubation period in humans",
    "league",
    "film crew member",
    "electric charge",
    "symptoms",
    "replaced by",
    "nominated for",
    "religion",
    "wavelength",
    "total produced",
    "time of discovery or invention",
    "invasive to",
    "use",
    "negative therapeutic predictor",
    "item operated",
    "participating team",
    "political ideology",
    "compulsory education (maximum age)",
    "applies to jurisdiction",
    "history of topic",
    "author",
    "mass",
    "heart rate",
    "killed by",
    "characters",
    "diocese",
    "Erdős number",
    "time period",
    "has part",
    "age of candidacy",
    "semi-major axis",
    "dual to",
    "official language",
    "production company",
    "replaced synonym (for nom. nov.)",
    "main regulatory text",
    "participant of",
    "head of government",
    "age of majority",
    "heritage designation",
    "drafted by",
    "family relationship degree",
    "discontinued date",
    "operator",
    "term length of office",
    "spin quantum number",
    "vehicles per capita (1000)",
    "enclave within",
    "embodied energy",
    "represents",
    "partner",
    "stepparent",
    "taxon synonym",
    "time of spacecraft launch",
    "conversion to standard unit",
    "nominal GDP",
    "lower flammable limit",
    "readable file format",
    "minimal incubation period in humans",
    "connecting line",
    "located in the administrative territorial entity",
    "place of burial",
    "contains administrative territorial entity",
    "statistical leader",
    "sports discipline competed in",
    "tensile modulus of elasticity",
    "research site",
    "connects with",
    "has cause",
    "date of birth",
    "location",
    "age of consent",
    "mains voltage",
    "industry",
    "basionym",
    "marriageable age",
    "visitors per year",
    "Poisson's ratio",
    "suicide rate",
    "carries scientific instrument",
    "connecting service",
    "place of detention",
    "crew member",
    "place served by transport hub",
    "organisation directed from the office or person",
    "memory capacity",
    "primary destinations",
    "relative permeability",
    "parent club",
    "organizer",
    "space launch vehicle",
    "encodes",
    "architect",
    "notable work",
    "commissioned by",
    "depicts",
    "individual tax rate",
    "website account on",
    "central bank",
    "software engine",
    "numeric value",
    "official religion",
    "wingspan",
    "occupation",
    "member count",
    "ceiling exposure limit",
    "date of first performance",
    "discoverer or inventor",
    "described by source",
    "executive body",
    "parent taxon",
    "pole position",
    "sports league level",
    "pKa",
    "genetic association",
    "mountain range",
    "part of",
    "legal form",
    "regulates (molecular biology)",
    "end time",
    "month of the year",
    "employer",
    "from fictional universe",
    "spouse",
    "copyright holder",
    "lake outflow",
    "solubility",
    "located in or next to body of water",
    "IDLH",
    "office held by head of the organisation",
    "office held by head of government",
    "territory claimed by",
    "tracklist",
    "takes place in fictional universe",
    "mount",
    "season of club or team",
    "this taxon is source of",
    "theme music",
    "Alexa rank",
    "film editor",
    "derivative work",
    "territory overlaps",
    "perimeter",
    "price",
    "secretary general",
    "frequency of event",
    "mascot",
    "maintained by",
    "duration",
    "screenwriter",
    "life expectancy",
    "minimum number of players",
    "winner",
    "native language",
    "start time",
    "highest point",
    "legislated by",
    "parity",
    "melting point",
    "location of formation",
    "ultimate tensile strength",
    "defined daily dose",
    "chief executive officer",
    "number of parts of this work of art",
    "endemic to",
    "subclass of",
    "dissolved, abolished or demolished",
    "service entry",
    "follows",
    "number of constituencies",
    "structural engineer",
    "writing system",
    "capital of",
    "taxonomic type",
    "next higher rank",
    "commemorates",
    "continent",
    "relative",
    "residence time of water",
    "number of speakers",
    "conferred by",
    "Gram staining",
    "work period (start)",
    "sport",
    "has effect",
    "tributary",
    "place of birth",
    "member of sports team",
    "relative permittivity",
    "instrument",
    "interested in",
    "academic degree",
    "location of discovery",
    "electronegativity",
    "located on terrain feature",
    "conflict",
    "height",
    "short-term exposure limit",
    "start point",
    "original language of film or TV show",
    "publication interval",
    "amended by",
    "material used",
    "located in present-day administrative territorial entity",
    "drainage basin",
    "lakes on river",
    "league level below",
    "licensed to broadcast to",
    "residence",
    "after a work by",
    "present in work",
    "basin country",
    "product certification",
    "mouth of the watercourse",
    "for work",
    "has quality",
    "uses",
    "time-weighted average exposure limit",
    "license",
    "significant person",
    "archives at",
    "natural product of taxon",
    "anthem",
    "adjacent station",
    "real gross domestic product growth rate",
    "carries",
    "member of political party",
    "professional or sports partner",
    "ethnic group",
    "member of",
    "platform",
    "destination point",
    "sports season of league or competition",
    "country for sport",
    "account charge / subscription fee",
    "patron saint",
    "compulsory education (minimum age)",
    "route of administration",
    "antiparticle",
    "sponsor",
    "floors above ground",
    "timezone offset",
    "programming language",
    "stock exchange",
    "opposite of",
    "mouthpiece",
    "unemployment rate",
    "watershed area",
    "editor",
    "collection",
    "award received",
    "designated as terrorist by",
    "illustrator",
    "student of",
    "dedicated to",
    "youth wing",
    "total fertility rate",
    "elevation above sea level",
    "repealed by",
    "practiced by",
    "named after",
    "movement",
    "flattening",
    "position played on team / speciality",
    "median lethal dose",
    "employees",
    "physically interacts with",
    "highway system",
    "parent peak",
    "participant",
    "number of cases",
    "editor-in-chief",
    "instance of",
    "sidekick of",
    "width",
    "cites",
    "child",
    "has edition",
    "doctoral student",
    "original network",
    "board member",
    "service retirement",
    "anatomical location",
    "biological variant of",
    "Euler characteristic",
    "diplomatic relation",
    "number of children",
    "narrative location",
    "incidence",
    "allegiance",
    "airline hub",
    "vapor pressure",
    "constellation",
    "voice actor",
    "number of platform tracks",
    "work period (end)",
    "military rank",
    "vertical depth",
    "vessel class",
    "parent astronomical body",
    "director/manager",
    "owner of",
    "distribution",
    "court",
    "angular resolution",
    "located on street",
    "owned by",
    "retirement age",
    "said to be the same as",
    "language used",
    "applies to part",
    "business division",
    "contains settlement",
    "main subject",
    "operating system",
    "authority",
    "number of representations",
    "ancestral home",
    "radius",
    "binding energy",
    "general manager",
    "measured by",
    "next lower rank",
    "cast member",
    "thermal conductivity",
    "health specialty",
    "father",
    "worshipped by",
    "headquarters location",
    "child astronomical body",
    "distributor",
    "noble title",
    "studied by",
    "officeholder",
    "genre",
    "vaccine for",
    "inception",
    "produced by",
    "narrator",
    "different from",
    "volcano observatory",
    "art director",
    "objective of project or action",
    "composer",
    "hardness",
    "edition or translation of",
    "isospin quantum number",
    "foundational text",
    "broadcast by",
    "office held by head of state",
    "boiling point",
    "minimum wavelength of sensitivity",
    "speaker",
    "studies",
    "capital",
    "terminus",
    "pressure",
    "number of episodes",
    "decomposition point",
    "filming location",
    "product or material produced",
    "gene inversion association with",
    "found in taxon",
    "field of work",
    "language of work or name",
    "ranking",
    "crosses",
    "culture",
    "location of first performance",
    "dialect of",
    "date of death",
    "influenced by",
]


class WebRedConfig(datasets.BuilderConfig):
    """BuilderConfig for WebRed."""

    def __init__(
        self,
        data_url,
        citation,
        url,
        class_labels,
        description,
        **kwargs,
    ):
        """BuilderConfig for WebRed.
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
class WebRed(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    BUILDER_CONFIGS = [
        WebRedConfig(
            name="webred_5",
            data_url="https://github.com/google-research-datasets/WebRED/raw/main/webred_5.tfrecord",
            citation=_CITATION_WEBRED,
            url=_HOMEPAGE,
            class_labels=_CLASS_LABELS,
            description=_DESCRIPTION
            + "\nEach example in WebRED 5 was annotated by exactly 5 independent human annotators.",
        ),
        WebRedConfig(
            name="webred_21",
            data_url="https://github.com/google-research-datasets/WebRED/raw/main/webred_21.tfrecord",
            citation=_CITATION_WEBRED,
            url=_HOMEPAGE,
            class_labels=_CLASS_LABELS,
            description=_DESCRIPTION
            + "\nIn WebRED 2+1, each example was annotated by 2 independent annotators. If they "
            "disagreed, an additional annotator (+1) was assigned to the example who also "
            "provided a disambiguating annotation.",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=self.config.description,
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
            citation=_CITATION_WEBRED,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        file_path = dl_manager.download_and_extract(self.config.data_url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": file_path},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        dataset = tf.data.TFRecordDataset(filepath)
        idx = 0
        nlp = English()

        def get_feature_value(feature, key):
            return feature[key].bytes_list.value[0].decode("utf-8")

        for raw_sentence in dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_sentence.numpy())

            rel_id = get_feature_value(example.features.feature, "relation_id")
            sentence = get_feature_value(example.features.feature, "sentence")
            label = get_feature_value(example.features.feature, "relation_name")

            # 1. Find OBJ{} and SUBJ{} marker indices
            subj = re.search("SUBJ{.+?}", sentence)
            obj = re.search("OBJ{.+?}", sentence)
            if not subj or not obj:
                print(f"Did not find OBJ or SUBJ marker in sentence: {sentence}")
                continue
            else:
                subj_start, subj_end = subj.span()
                obj_start, obj_end = obj.span()
            # 2. OPTIONAL: Replace with source and target strings (they contain special characters while the sentence
            # contains standard writing?)
            # source = get_feature_value(sentence.features.feature, "source_name")
            # target = get_feature_value(sentence.features.feature, "target_name")

            # 3. Remove markers and adjust indices: divide sentence at marker indices, remove marker, merge
            # what if subj or obj is at the start or end of the sentence?
            cleaned_sentence = ""
            if subj_start < obj_start:
                cleaned_sentence += sentence[:subj_start]
                cleaned_sentence += sentence[subj_start + 5 : subj_end - 1]
                cleaned_sentence += sentence[subj_end:obj_start]
                cleaned_sentence += sentence[obj_start + 4 : obj_end - 1]
                cleaned_sentence += sentence[obj_end:]
                subj_end -= 6
                obj_start -= 6
                obj_end -= 11
            else:
                cleaned_sentence += sentence[:obj_start]
                cleaned_sentence += sentence[obj_start + 4 : obj_end - 1]
                cleaned_sentence += sentence[obj_end:subj_start]
                cleaned_sentence += sentence[subj_start + 5 : subj_end - 1]
                cleaned_sentence += sentence[subj_end:]
                obj_end -= 5
                subj_start -= 5
                subj_end -= 11
            # 4. Tokenize and calculate token indices from char offsets
            doc = nlp(cleaned_sentence)
            tokens = [token.text for token in doc]
            subj_span = doc.char_span(subj_start, subj_end, alignment_mode="expand")
            head_start = subj_span.start
            head_end = subj_span.end
            obj_span = doc.char_span(obj_start, obj_end, alignment_mode="expand")
            tail_start = obj_span.start
            tail_end = obj_span.end

            id_ = str(idx) + "_" + rel_id
            idx += 1

            yield id_, {
                "tokens": tokens,
                "head_start": head_start,
                "head_end": head_end,
                "tail_start": tail_start,
                "tail_end": tail_end,
                "label": label,
            }
