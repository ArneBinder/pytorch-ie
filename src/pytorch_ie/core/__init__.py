import sys

import pie_core
from pie_core.document import Annotation, AnnotationLayer, Document, annotation_field
from pie_core.metric import DocumentMetric
from pie_core.module_mixins import (
    EnterDatasetDictMixin,
    EnterDatasetMixin,
    ExitDatasetDictMixin,
    ExitDatasetMixin,
    PreparableMixin,
    WithDocumentTypeMixin,
)
from pie_core.statistic import DocumentStatistic
from pie_core.taskmodule import AutoTaskModule, TaskEncoding, TaskModule

from .model import AutoModel, PyTorchIEModel

submodules = ["document", "taskmodule", "metric", "statistic"]
for sub in submodules:
    module = getattr(pie_core, sub)
    sys.modules[f"{__name__}.{sub}"] = module

# backwards compatibility
AnnotationList = AnnotationLayer
RequiresDocumentTypeMixin = WithDocumentTypeMixin
