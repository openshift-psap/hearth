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


class GPUDiscoveryError(Exception):
    pass


@dataclass(frozen=True)
class DiscoveredGPU:
    vendor: str
    model: str
    short_name: str
    count: int
    node_count: int


@dataclass(frozen=True)
class DiscoveryResult:
    gpus: tuple[DiscoveredGPU, ...]
    total_gpus: int
    timestamp: str


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
            normalized = normalized[len(prefix):]
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
            nodes = target_core.list_node(_request_timeout=timeout)
        except Exception as exc:
            raise GPUDiscoveryError(
                f"Failed to connect to cluster {cluster_name!r}: {exc}"
            ) from exc
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

        logger.info(
            "Discovered %d GPUs on cluster %s: %s",
            total,
            cluster_name,
            ", ".join(f"{g.count}x {g.short_name}" for g in gpus) or "none",
        )

        return DiscoveryResult(gpus=gpus, total_gpus=total, timestamp=timestamp)
