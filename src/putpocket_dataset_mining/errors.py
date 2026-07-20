from __future__ import annotations


class DatasetMiningError(Exception):
    """Base error for dataset-mining failures."""


class ConfigError(DatasetMiningError):
    """Raised for invalid configuration."""


class DependencyError(DatasetMiningError):
    """Raised when a required runtime dependency is unavailable."""


class InfraError(DatasetMiningError):
    """Raised for Docker, vLLM, filesystem, or CLI infrastructure failures."""


class ToolParseError(DatasetMiningError):
    """Raised when a model response cannot be parsed as a Cline tool call."""
