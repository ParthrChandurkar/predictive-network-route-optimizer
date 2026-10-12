"""Small, opt-in OpenTelemetry facade used by routing and experiments."""
from __future__ import annotations

import functools
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable


def _enabled_from_environment() -> bool:
    value = os.getenv("OBSERVABILITY_ENABLED", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _clean_endpoint(value: str) -> str:
    return value.rstrip("/")


@dataclass(frozen=True)
class ObservabilityState:
    enabled: bool
    service_name: str
    endpoint: str | None


class _NoOpTelemetry:
    state = ObservabilityState(False, "predictive-network-route-optimizer", None)

    def route(self, result: dict[str, Any], risk_aware: bool) -> None:
        return None

    def route_span(self, risk_aware: bool):
        return nullcontext(None)

    def inference(self, row_count: int, elapsed_ms: float, scores: Any) -> None:
        return None

    def inference_span(self, row_count: int):
        return nullcontext(None)

    def mininet_trial(self, trial: dict[str, Any]) -> None:
        return None

    def mininet_span(self, route_name: str):
        return nullcontext(None)

    def shutdown(self) -> None:
        return None


class _OpenTelemetry:
    def __init__(self, service_name: str, endpoint: str, export_interval_ms: int):
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": "phase-5",
                "deployment.environment.name": os.getenv("DEPLOYMENT_ENVIRONMENT", "local"),
            }
        )
        self._tracer_provider = TracerProvider(resource=resource)
        self._tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        )
        self._metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"),
            export_interval_millis=export_interval_ms,
        )
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[self._metric_reader],
        )
        trace.set_tracer_provider(self._tracer_provider)
        metrics.set_meter_provider(self._meter_provider)
        self._tracer = trace.get_tracer("network-route-optimizer")
        self._trace_api = trace
        meter = metrics.get_meter("network-route-optimizer")
        self._route_count = meter.create_counter(
            "route.calculations", description="Number of route calculations"
        )
        self._route_latency = meter.create_histogram(
            "route.duration", unit="ms", description="Route calculation duration"
        )
        self._route_hops = meter.create_histogram("route.hops", unit="{hop}")
        self._route_risk = meter.create_histogram("route.average_risk", unit="1")
        self._inference_count = meter.create_counter("model.inference.rows", unit="{row}")
        self._inference_latency = meter.create_histogram("model.inference.duration", unit="ms")
        self._inference_score = meter.create_histogram("model.degradation_score", unit="1")
        self._experiment_throughput = meter.create_histogram(
            "mininet.throughput", unit="Mbit/s"
        )
        self._experiment_rtt = meter.create_histogram("mininet.rtt", unit="ms")
        self._experiment_loss = meter.create_histogram("mininet.packet_loss", unit="%")
        self.state = ObservabilityState(True, service_name, endpoint)

    def route_span(self, risk_aware: bool):
        return self._tracer.start_as_current_span(
            "route.calculate", attributes={"route.risk_aware": risk_aware}
        )

    def route(self, result: dict[str, Any], risk_aware: bool) -> None:
        attributes = {
            "route.algorithm": str(result.get("algorithm", "unknown")),
            "route.risk_aware": risk_aware,
            "route.found": bool(result.get("found", False)),
        }
        span = self._trace_api.get_current_span()
        if span.is_recording():
            for name, value in attributes.items():
                span.set_attribute(name, value)
            span.set_attribute("route.hops", int(result.get("hops", 0)))
            span.set_attribute("route.average_risk", float(result.get("avg_risk", 0.0)))
            span.set_attribute("route.path", str(result.get("path_str", "")))
        self._route_count.add(1, attributes)
        self._route_latency.record(float(result.get("elapsed_ms", 0.0)), attributes)
        self._route_hops.record(int(result.get("hops", 0)), attributes)
        self._route_risk.record(float(result.get("avg_risk", 0.0)), attributes)

    def inference(self, row_count: int, elapsed_ms: float, scores: Any) -> None:
        attributes = {"model.name": "lte-degradation-random-forest"}
        span = self._trace_api.get_current_span()
        if span.is_recording():
            span.set_attribute("model.duration_ms", elapsed_ms)
        self._inference_count.add(row_count, attributes)
        self._inference_latency.record(elapsed_ms, attributes)
        for score in scores:
            self._inference_score.record(float(score), attributes)

    def inference_span(self, row_count: int):
        return self._tracer.start_as_current_span(
            "model.predict_lte_risk",
            attributes={
                "model.name": "lte-degradation-random-forest",
                "model.rows": row_count,
            },
        )

    def mininet_trial(self, trial: dict[str, Any]) -> None:
        attributes = {"experiment.route": str(trial["route"])}
        span = self._trace_api.get_current_span()
        throughput = float(trial["iperf"]["throughput_mbps"])
        loss = float(trial["ping"]["packet_loss_percent"])
        rtt = trial["ping"].get("average_rtt_ms")
        if span.is_recording():
            span.set_attribute("network.throughput_mbps", throughput)
            span.set_attribute("network.packet_loss_percent", loss)
            if rtt is not None:
                span.set_attribute("network.rtt_ms", float(rtt))
        self._experiment_throughput.record(throughput, attributes)
        self._experiment_loss.record(loss, attributes)
        if rtt is not None:
            self._experiment_rtt.record(float(rtt), attributes)

    def mininet_span(self, route_name: str):
        return self._tracer.start_as_current_span(
            "mininet.trial", attributes={"experiment.route": route_name}
        )

    def shutdown(self) -> None:
        self._meter_provider.shutdown()
        self._tracer_provider.shutdown()


_LOCK = threading.Lock()
_TELEMETRY: _NoOpTelemetry | _OpenTelemetry = _NoOpTelemetry()


def configure_observability(
    enabled: bool | None = None,
    service_name: str | None = None,
    endpoint: str | None = None,
    export_interval_ms: int = 5000,
) -> ObservabilityState:
    """Configure OTLP/HTTP export once; remain a no-op unless explicitly enabled."""
    global _TELEMETRY
    should_enable = _enabled_from_environment() if enabled is None else enabled
    if not should_enable:
        return _TELEMETRY.state
    with _LOCK:
        if _TELEMETRY.state.enabled:
            return _TELEMETRY.state
        resolved_name = service_name or os.getenv(
            "OTEL_SERVICE_NAME", "predictive-network-route-optimizer"
        )
        resolved_endpoint = _clean_endpoint(
            endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        )
        _TELEMETRY = _OpenTelemetry(
            resolved_name, resolved_endpoint, export_interval_ms
        )
        return _TELEMETRY.state


def observe_route(risk_aware: bool) -> Callable:
    """Decorate a route function without changing its public API."""
    def decorator(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with _TELEMETRY.route_span(risk_aware):
                result = function(*args, **kwargs)
                _TELEMETRY.route(result, risk_aware)
                return result

        return wrapped

    return decorator


def record_model_inference(row_count: int, started_at: float, scores: Any) -> None:
    _TELEMETRY.inference(row_count, (time.perf_counter() - started_at) * 1000, scores)


def model_inference_span(row_count: int):
    return _TELEMETRY.inference_span(row_count)


def observe_mininet_trial(function: Callable) -> Callable:
    """Trace a complete Mininet trial and record its resulting measurements."""
    @functools.wraps(function)
    def wrapped(plan: dict, route_name: str, *args, **kwargs):
        with _TELEMETRY.mininet_span(route_name):
            result = function(plan, route_name, *args, **kwargs)
            _TELEMETRY.mininet_trial(result)
            return result

    return wrapped


def shutdown_observability() -> None:
    _TELEMETRY.shutdown()
