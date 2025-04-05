import sys

import pie_core
from pie_core.document import Annotation, AnnotationLayer, Document, annotation_field
from pie_core.metric import DocumentMetric
from pie_core.module_mixins import (
    EnterDatasetDictMixin,
    EnterDatasetMixin,
    ExitDatasetDictMixin,
    ExitDatasetMixin,
    WithDocumentTypeMixin,
)
from pie_core.preparable import PreparableMixin
from pie_core.statistic import DocumentStatistic
from pie_core.taskmodule import AutoTaskModule, TaskEncoding, TaskModule

from pytorch_ie import model
from pytorch_ie.model import AutoModel, PyTorchIEModel

submodules = ["document", "taskmodule", "metric", "statistic"]
for sub in submodules:
    module = getattr(pie_core, sub)
    sys.modules[f"{__name__}.{sub}"] = module

sys.modules[f"{__name__}.model"] = model

# backwards compatibility
AnnotationList = AnnotationLayer
RequiresDocumentTypeMixin = WithDocumentTypeMixin
