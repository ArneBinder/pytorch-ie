from .document import Annotation, AnnotationLayer, Document, annotation_field, Comparable
from .metric import DocumentMetric
from .model import PyTorchIEModel
from .module_mixins import (
    EnterDatasetDictMixin,
    EnterDatasetMixin,
    ExitDatasetDictMixin,
    ExitDatasetMixin,
    PreparableMixin,
    WithDocumentTypeMixin,
)
from .statistic import DocumentStatistic
from .taskmodule import TaskEncoding, TaskModule

# backwards compatibility
AnnotationList = AnnotationLayer
RequiresDocumentTypeMixin = WithDocumentTypeMixin
