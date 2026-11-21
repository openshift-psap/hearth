from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import yaml
from kubernetes import client, config as k8s_config

from hearth.settings import settings

logger = logging.getLogger(__name__)


GPU_MODEL_MAP: dict[str, str] = {
    "NVIDIA-A100-SXM4-80GB": "a100",
    "NVIDIA-A100-SXM4-40GB": "a100",
    "NVIDIA-A100-PCIE-80GB": "a100",
    "NVIDIA-A100-PCIE-40GB": "a100",
    "NVIDIA-H100-SXM5-80GB": "h100",
    "NVIDIA-H100-PCIE-80GB": "h100",
    "NVIDIA-H200-SXM-141GB": "h200",
    "AMD-Instinct-MI300X": "mi300x",
}

GPU_RESOURCE_NVIDIA = "nvidia.com/gpu"
GPU_RESOURCE_AMD = "amd.com/gpu"
GPU_LABEL_NVIDIA = "nvidia.com/gpu.product"
GPU_LABEL_AMD = "amd.com/gpu.product"
GPU_RESOURCES = (GPU_RESOURCE_NVIDIA, GPU_RESOURCE_AMD)


class GPUDiscoveryError(Exception):
    pass


def _quantity(value: object) -> int:
    """Read a GPU resource quantity as an integer, tolerating malformed input."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _node_ready(node: client.V1Node) -> bool:
    """Whether the node reports Ready=True.

    A missing Ready condition is not evidence of readiness, so the node is
    treated as not ready and contributes no free capacity.
    """
    conditions = getattr(node.status, "conditions", None)
    try:
        conditions = list(conditions or ())
    except TypeError:
        return False
    for condition in conditions:
        if getattr(condition, "type", None) == "Ready":
            return str(getattr(condition, "status", "")).lower() == "true"
    return False


def _node_unschedulable(node: client.V1Node) -> bool:
    return getattr(node.spec, "unschedulable", False) is True


def _container_gpu_requests(container: client.V1Container) -> dict[str, int]:
    resources = getattr(container, "resources", None)
    requests = getattr(resources, "requests", None) if resources is not None else None
    if not isinstance(requests, dict):
        return {}
    counts: dict[str, int] = {}
    for resource in GPU_RESOURCES:
        count = _quantity(requests.get(resource, 0))
        if count > 0:
            counts[resource] = count
    return counts


def _sum_requests(containers) -> dict[str, int]:
    totals: dict[str, int] = {}
    for container in containers:
        totals = _add_requests(totals, _container_gpu_requests(container))
    return totals


def _add_requests(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    result = dict(left)
    for resource, count in right.items():
        result[resource] = result.get(resource, 0) + count
    return result


def _max_requests(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    result = dict(left)
    for resource, count in right.items():
        result[resource] = max(result.get(resource, 0), count)
    return result


def _is_sidecar(container: client.V1Container) -> bool:
    """A restartable init container runs alongside the app containers."""
    return getattr(container, "restart_policy", None) == "Always"


def _pod_gpu_requests(pod: client.V1Pod) -> dict[str, int]:
    """GPUs held by a pod, per resource name.

    Follows Kubernetes' effective pod request: sidecars start in order among
    the init containers, each regular init container runs together with the
    sidecars started before it, and the app containers run with all sidecars.
    The pod holds the largest of those totals.
    """
    spec = getattr(pod, "spec", None)
    if spec is None:
        return {}
    try:
        containers = list(getattr(spec, "containers", None) or ())
        init_containers = list(getattr(spec, "init_containers", None) or ())
    except TypeError:
        return {}

    started_sidecars: dict[str, int] = {}
    init_totals: dict[str, int] = {}
    for container in init_containers:
        requests = _container_gpu_requests(container)
        if _is_sidecar(container):
            started_sidecars = _add_requests(started_sidecars, requests)
            continue
        step = _add_requests(started_sidecars, requests)
        init_totals = _max_requests(init_totals, step)

    app_totals = _add_requests(_sum_requests(containers), started_sidecars)
    return _max_requests(app_totals, init_totals)


@dataclass(frozen=True)
class DiscoveredGPU:
    vendor: str
    model: str
    short_name: str
    count: int
    node_count: int


@dataclass(frozen=True)
class NodeGPUCapacity:
    """Total and currently free GPUs on one target node.

    ``free`` is zero for a node that is not Ready or is cordoned, so a caller
    never schedules onto it. ``DiscoveryResult.nodes`` is empty when pod usage
    could not be read at all; an empty tuple means "unknown", not "no capacity".
    """

    name: str
    short_name: str
    total: int
    free: int


@dataclass(frozen=True)
class DiscoveryResult:
    gpus: tuple[DiscoveredGPU, ...]
    total_gpus: int
    timestamp: str
    nodes: tuple[NodeGPUCapacity, ...] = ()


def _normalize_gpu_model(raw_model: str) -> str:
    if raw_model in GPU_MODEL_MAP:
        return GPU_MODEL_MAP[raw_model]
    for known, short in GPU_MODEL_MAP.items():
        base = "-".join(known.split("-")[:2])
        if raw_model.startswith(base):
            return short
    # Fallback: strip vendor prefix and non-alphanumeric chars for CRD compatibility.
    # Examples: NVIDIA-L40S -> l40s, NVIDIA-B200-SXM -> b200sxm, NVIDIA-A10G -> a10g
    normalized = raw_model.lower()
    for prefix in ("nvidia-", "amd-instinct-", "amd-"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return re.sub(r"[^a-z0-9]", "", normalized)


class GPUDiscoveryClient:
    def __init__(self, management_core: client.CoreV1Api) -> None:
        self._management_core = management_core

    def _read_kubeconfig_from_secret(self, secret_name: str, namespace: str) -> dict:
        try:
            secret = self._management_core.read_namespaced_secret(secret_name, namespace)
        except client.exceptions.ApiException as exc:
            if exc.status == 404:
                raise GPUDiscoveryError(
                    f"Kubeconfig secret {secret_name!r} not found in {namespace}"
                ) from exc
            raise GPUDiscoveryError(
                f"Failed to read kubeconfig secret {secret_name!r} from {namespace}: {exc}"
            ) from exc

        raw = None
        if secret.data and "kubeconfig" in secret.data:
            try:
                raw = base64.b64decode(secret.data["kubeconfig"]).decode()
            except Exception as exc:
                raise GPUDiscoveryError(
                    f"Secret {secret_name!r} has invalid base64 kubeconfig data"
                ) from exc

        if not raw:
            raise GPUDiscoveryError(f"Secret {secret_name!r} has no 'kubeconfig' key")

        try:
            parsed = yaml.safe_load(raw)
        except Exception as exc:
            raise GPUDiscoveryError(f"Secret {secret_name!r} kubeconfig is not valid YAML") from exc

        if isinstance(parsed, str):
            try:
                parsed = yaml.safe_load(base64.b64decode(parsed).decode())
            except Exception as exc:
                logger.debug(
                    "Secret %r: double-base64 decode failed, will attempt as raw YAML: %s",
                    secret_name,
                    exc,
                )

        if not isinstance(parsed, dict):
            raise GPUDiscoveryError(f"Secret {secret_name!r} kubeconfig is not valid YAML")
        return parsed

    def discover_gpus(
        self,
        cluster_name: str,
        kubeconfig_secret: str,
        secrets_namespace: str,
    ) -> DiscoveryResult:
        kubeconfig_dict = self._read_kubeconfig_from_secret(kubeconfig_secret, secrets_namespace)

        if "current-context" not in kubeconfig_dict:
            contexts = kubeconfig_dict.get("contexts", [])
            if contexts:
                kubeconfig_dict["current-context"] = contexts[0]["name"]
            else:
                raise GPUDiscoveryError(f"Kubeconfig for {cluster_name!r} has no contexts")

        try:
            api_client = k8s_config.new_client_from_config_dict(kubeconfig_dict)
        except Exception as exc:
            raise GPUDiscoveryError(
                f"Invalid kubeconfig for cluster {cluster_name!r}: {exc}"
            ) from exc
        timeout = (settings.gpu_discovery_timeout_sec, settings.gpu_discovery_timeout_sec)

        try:
            target_core = client.CoreV1Api(api_client)
            try:
                nodes = target_core.list_node(_request_timeout=timeout)
            except Exception as exc:
                raise GPUDiscoveryError(
                    f"Failed to connect to cluster {cluster_name!r}: {exc}"
                ) from exc
            try:
                pods = target_core.list_pod_for_all_namespaces(_request_timeout=timeout)
                pod_items: list[client.V1Pod] | None = list(pods.items or ())
            except Exception as exc:  # noqa: BLE001 - pod listing is best effort
                logger.warning(
                    "Cluster %s: could not list pods, free GPU blocks unavailable: %s",
                    cluster_name,
                    exc,
                )
                pod_items = None
        finally:
            api_client.close()

        gpu_counts: dict[tuple[str, str, str], list[int]] = {}

        for node in nodes.items:
            allocatable = node.status.allocatable or {}
            labels = node.metadata.labels or {}

            nvidia_count = int(allocatable.get(GPU_RESOURCE_NVIDIA, 0))
            amd_count = int(allocatable.get(GPU_RESOURCE_AMD, 0))

            if nvidia_count > 0:
                raw_model = labels.get(GPU_LABEL_NVIDIA, "")
                short = _normalize_gpu_model(raw_model) if raw_model else "nvidia"
                key = ("nvidia", raw_model or "nvidia", short)
                gpu_counts.setdefault(key, []).append(nvidia_count)

            if amd_count > 0:
                raw_model = labels.get(GPU_LABEL_AMD, "")
                short = _normalize_gpu_model(raw_model) if raw_model else "amd"
                key = ("amd", raw_model or "amd", short)
                gpu_counts.setdefault(key, []).append(amd_count)

        gpus = tuple(
            DiscoveredGPU(
                vendor=vendor,
                model=model,
                short_name=short,
                count=sum(counts),
                node_count=len(counts),
            )
            for (vendor, model, short), counts in sorted(gpu_counts.items())
        )

        total = sum(g.count for g in gpus)
        timestamp = datetime.now(timezone.utc).isoformat()

        node_capacities: tuple[NodeGPUCapacity, ...] = ()
        if pod_items is not None:
            usage: dict[str, dict[str, int]] = {}
            for pod in pod_items:
                pod_spec = getattr(pod, "spec", None)
                node_name = getattr(pod_spec, "node_name", None)
                phase = getattr(getattr(pod, "status", None), "phase", None)
                if not node_name or phase in ("Succeeded", "Failed"):
                    continue
                usage[node_name] = _add_requests(usage.get(node_name, {}), _pod_gpu_requests(pod))

            blocks: list[NodeGPUCapacity] = []
            for node in nodes.items:
                allocatable = node.status.allocatable or {}
                labels = node.metadata.labels or {}
                node_usage = usage.get(node.metadata.name, {})
                schedulable = _node_ready(node) and not _node_unschedulable(node)
                # Use the same short name discovery assigns, so the annotation
                # key matches the GPU type Fournos validates.
                for resource, label, fallback in (
                    (GPU_RESOURCE_NVIDIA, GPU_LABEL_NVIDIA, "nvidia"),
                    (GPU_RESOURCE_AMD, GPU_LABEL_AMD, "amd"),
                ):
                    node_total = int(allocatable.get(resource, 0))
                    if node_total <= 0:
                        continue
                    raw_model = labels.get(label, "")
                    short = _normalize_gpu_model(raw_model) if raw_model else fallback
                    free = 0
                    if schedulable:
                        free = max(0, node_total - node_usage.get(resource, 0))
                    blocks.append(
                        NodeGPUCapacity(
                            name=node.metadata.name,
                            short_name=short,
                            total=node_total,
                            free=free,
                        )
                    )
            node_capacities = tuple(
                sorted(blocks, key=lambda block: (block.name, block.short_name))
            )

        logger.info(
            "Discovered %d GPUs on cluster %s: %s",
            total,
            cluster_name,
            ", ".join(f"{g.count}x {g.short_name}" for g in gpus) or "none",
        )

        return DiscoveryResult(
            gpus=gpus,
            total_gpus=total,
            timestamp=timestamp,
            nodes=node_capacities,
        )
