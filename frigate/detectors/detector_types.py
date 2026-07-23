import importlib
import logging
import pkgutil
from enum import Enum
from typing import Annotated, Union

from pydantic import Field

from . import plugins
from .detection_api import DetectionApi
from .detector_config import BaseDetectorConfig

logger = logging.getLogger(__name__)


_included_modules = pkgutil.iter_modules(plugins.__path__, plugins.__name__ + ".")

plugin_modules = []

for _, name, _ in _included_modules:
    try:
        # currently openvino may fail when importing
        # on an arm device with 64 KiB page size.
        plugin_modules.append(importlib.import_module(name))
    except ImportError as e:
        logger.error(f"Error importing detector runtime: {e}")


api_types = {det.type_key: det for det in DetectionApi.__subclasses__()}


def detector_supports_multiple_models(type_key: str) -> bool:
    """Return whether the given detector type can run multiple model instances."""
    detector = api_types.get(type_key)
    return bool(detector and getattr(detector, "supports_multiple_models", False))


class StrEnum(str, Enum):
    pass


DetectorTypeEnum = StrEnum("DetectorTypeEnum", {k: k for k in api_types})

DetectorConfig = Annotated[
    Union[tuple(BaseDetectorConfig.__subclasses__())],  # noqa: UP007
    Field(discriminator="type"),
]
