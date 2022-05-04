# flake8: noqa

from auto import AutoModel, AutoPipeline, AutoTaskModule

from pytorch_ie.annotations import (
    BinaryRelation,
    Label,
    LabeledMultiSpan,
    LabeledSpan,
    MultiLabel,
    MultiLabeledBinaryRelation,
    MultiLabeledMultiSpan,
    MultiLabeledSpan,
    Span,
)
from pytorch_ie.core.document import Annotation, AnnotationList, Document, annotation_field
from pytorch_ie.core.model import PyTorchIEModel
from pytorch_ie.core.taskmodule import TaskModule
from pytorch_ie.data import *
from pytorch_ie.documents import TextDocument
from pytorch_ie.pipeline import Pipeline
