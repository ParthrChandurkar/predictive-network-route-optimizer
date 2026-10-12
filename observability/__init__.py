"""Optional OpenTelemetry instrumentation for the project."""

from observability.telemetry import (
    configure_observability,
    model_inference_span,
    observe_mininet_trial,
    observe_route,
    record_model_inference,
    shutdown_observability,
)

__all__ = [
    "configure_observability",
    "model_inference_span",
    "observe_mininet_trial",
    "observe_route",
    "record_model_inference",
    "shutdown_observability",
]
