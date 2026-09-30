"""OpenVINO runtime discovery without importing it during package startup."""

from __future__ import annotations

from typing import Any


class RuntimeUnavailableError(RuntimeError):
    pass


def probe_openvino(requested_device: str = "GPU") -> dict[str, Any]:
    try:
        import openvino as ov
    except ImportError as error:
        raise RuntimeUnavailableError(
            "OpenVINO is not installed. Install the PTL-supported OpenVINO release first."
        ) from error

    core = ov.Core()
    available_devices = list(core.available_devices)
    device = _resolve_device(requested_device, available_devices)
    supported_properties = {
        str(value) for value in core.get_property(device, "SUPPORTED_PROPERTIES")
    }

    properties: dict[str, Any] = {}
    for name in (
        "FULL_DEVICE_NAME",
        "DEVICE_ARCHITECTURE",
        "OPTIMIZATION_CAPABILITIES",
        "RANGE_FOR_ASYNC_INFER_REQUESTS",
        "RANGE_FOR_STREAMS",
        "GPU_DEVICE_TOTAL_MEM_SIZE",
        "GPU_EXECUTION_UNITS_COUNT",
    ):
        if name in supported_properties:
            properties[name.lower()] = _json_safe(core.get_property(device, name))

    return {
        "openvino_version": getattr(ov, "__version__", "unknown"),
        "requested_device": requested_device,
        "selected_device": device,
        "available_devices": available_devices,
        "properties": properties,
    }


def _resolve_device(requested: str, available: list[str]) -> str:
    if requested in available:
        return requested
    if requested == "GPU" and "GPU.0" in available:
        return "GPU.0"
    raise RuntimeUnavailableError(
        f"Requested device {requested!r} is unavailable; found {available or 'no devices'}"
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    return str(value)
