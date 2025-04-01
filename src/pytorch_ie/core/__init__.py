# from pie_core import auto, document, metric, model, module_mixins, statistic, taskmodule
from pie_core.auto import AutoModel, AutoTaskModule
from pie_core.document import Annotation, AnnotationLayer, Document, annotation_field
from pie_core.metric import DocumentMetric
from pie_core.model import PyTorchIEModel
from pie_core.module_mixins import (
    EnterDatasetDictMixin,
    EnterDatasetMixin,
    ExitDatasetDictMixin,
    ExitDatasetMixin,
    PreparableMixin,
    WithDocumentTypeMixin,
)
from pie_core.statistic import DocumentStatistic
from pie_core.taskmodule import TaskEncoding, TaskModule

# backwards compatibility
AnnotationList = AnnotationLayer
RequiresDocumentTypeMixin = WithDocumentTypeMixin
