"""Framework-specific exceptions with actionable failure boundaries."""


class SolarWMError(RuntimeError):
    """Base class for expected framework failures."""


class BackendContractError(SolarWMError):
    """The v4 reference conditioners cannot satisfy their runtime contract."""
