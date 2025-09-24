class RequiresModelNameOrPath:
    pass


class RequiresNumClasses:
    pass


class RequiresMaxInputLength:
    """Any class inheriting from this class should require a constructor parameter
    'max_input_length'."""

    pass


class RequiresTaskmoduleConfig:
    """Any class inheriting from this class should require a constructor parameter
    'taskmodule_config'."""

    pass
