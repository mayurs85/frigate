import logging

from .detector_config import InputTensorEnum, ModelConfig, PixelFormatEnum  # noqa: F401
from .detector_types import (  # noqa: F401
    DetectorConfig,
    DetectorTypeEnum,
    api_types,
    detector_supports_multiple_models,
)

logger = logging.getLogger(__name__)


def assign_detector_instances(
    detector_types: dict[str, str],
    used_models: list[str],
) -> list[tuple[str, str, str]]:
    """Assign detector entries to models.

    Detector types that support multiple models get one instance per model.
    Single-model detector entries are round-robin assigned across the models,
    wrapping around so that every detector entry is assigned.

    Args:
        detector_types: Detector key to detector type, in config order
        used_models: Ordered model keys in use by cameras

    Returns:
        List of (instance_name, detector_key, model_key) assignments
    """
    multi = [
        key
        for key, type_key in detector_types.items()
        if detector_supports_multiple_models(type_key)
    ]
    single = [key for key in detector_types if key not in multi]

    if not multi and len(single) < len(used_models):
        single_types = sorted({detector_types[key] for key in single})
        raise ValueError(
            f"Detectors {', '.join(single)} (types: {', '.join(single_types)}) can each only run a single model, "
            f"but {len(used_models)} models are in use ({', '.join(used_models)}). "
            "Add more detectors, use a detector type that supports multiple models, or reduce the number of models assigned to cameras."
        )

    assignments: list[tuple[str, str, str]] = []

    for key in multi:
        for model_key in used_models:
            instance_name = key if len(used_models) == 1 else f"{key}_{model_key}"
            assignments.append((instance_name, key, model_key))

    for i, key in enumerate(single):
        assignments.append((key, key, used_models[i % len(used_models)]))

    instance_names = [name for name, _, _ in assignments]
    duplicates = {name for name in instance_names if instance_names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"Detector instance names collide: {', '.join(sorted(duplicates))}. "
            "Rename the conflicting detectors or models so that expanded instance names (detector_model) are unique."
        )

    return assignments


def create_detector(detector_config):
    if detector_config.type == DetectorTypeEnum.cpu:
        logger.warning(
            "CPU detectors are not recommended and should only be used for testing or for trial purposes."
        )

    api = api_types.get(detector_config.type)
    if not api:
        raise ValueError(detector_config.type)
    return api(detector_config)
