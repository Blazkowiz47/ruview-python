"""Core exception hierarchy for the Python research port."""


class RuViewError(Exception):
    """Base class for RuView Python errors."""


class ValidationError(RuViewError, ValueError):
    """Raised when a data contract receives invalid input."""

