from typing import Any, Dict, Optional

from pie_core import AutoTaskModule, TaskModule

from ..interface import RequiresTaskmoduleConfig


class HasTaskmodule(RequiresTaskmoduleConfig):
    """A mixin class for models that have a taskmodule.

    Args:
        taskmodule_config: The config for the taskmodule which can be obtained from the
            taskmodule.config property.
    """

    def __init__(self, taskmodule_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.taskmodule: Optional[TaskModule] = None
        if taskmodule_config is not None:
            self.taskmodule = AutoTaskModule.from_config(taskmodule_config)
