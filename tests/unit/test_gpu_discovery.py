"""Tests for GPU discovery client."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from hearth.core.gpu_discovery import (
    GPUDiscoveryClient,
    GPUDiscoveryError,
    DiscoveredGPU,
    DiscoveryResult,
    _normalize_gpu_model,
)


class TestNormalizeGPUModel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("NVIDIA-A100-SXM4-80GB", "a100"),
            ("NVIDIA-A100-SXM4-40GB", "a100"),
            ("NVIDIA-A100-PCIE-80GB", "a100"),
            ("NVIDIA-H100-SXM5-80GB", "h100"),
            ("NVIDIA-H200-SXM-141GB", "h200"),
            ("AMD-Instinct-MI300X", "mi300x"),
        ],
    )
    def test_known_models(self, raw: str, expected: str) -> None:
        assert _normalize_gpu_model(raw) == expected

    def test_prefix_fallback(self) -> None:
        result = _normalize_gpu_model("NVIDIA-A100-NEW-VARIANT-96GB")
        assert result == "a100"

    def test_nvidia_l40s_fallback(self) -> None:
        assert _normalize_gpu_model("NVIDIA-L40S") == "l40s"

    def test_nvidia_b200_fallback(self) -> None:
        assert _normalize_gpu_model("NVIDIA-B200-SXM") == "b200sxm"

    def test_nvidia_a10g_fallback(self) -> None:
        assert _normalize_gpu_model("NVIDIA-A10G") == "a10g"

    def test_unknown_model_alphanumeric(self) -> None:
        result = _normalize_gpu_model("SOME-UNKNOWN-GPU")
        assert result == "someunknowngpu"

    def test_unknown_with_spaces(self) -> None:
        result = _normalize_gpu_model("NVIDIA Tesla V100")
        assert result == "nvidiateslav100"


class _DiscoveryTestHelpers:
    def _make_client(self) -> tuple[GPUDiscoveryClient, MagicMock]:
        mock_core = MagicMock()
        return GPUDiscoveryClient(mock_core), mock_core

    def _make_node(
        self,
        name: str,
        nvidia_gpus: int = 0,
        amd_gpus: int = 0,
        nvidia_model: str = "NVIDIA-A100-SXM4-80GB",
        amd_model: str = "AMD-Instinct-MI300X",
        ready: bool = True,
        unschedulable: bool = False,
    ) -> MagicMock:
        node = MagicMock()
        node.metadata.name = name
        allocatable = {}
        labels = {}

        if nvidia_gpus > 0:
            allocatable["nvidia.com/gpu"] = str(nvidia_gpus)
            labels["nvidia.com/gpu.product"] = nvidia_model
        if amd_gpus > 0:
            allocatable["amd.com/gpu"] = str(amd_gpus)
            labels["amd.com/gpu.product"] = amd_model

        condition = MagicMock()
        condition.type = "Ready"
        condition.status = "True" if ready else "False"
        node.status.conditions = [condition]
        node.status.allocatable = allocatable
        node.metadata.labels = labels
        node.spec.unschedulable = unschedulable
        return node

    def _mock_kubeconfig_secret(self, mock_core: MagicMock) -> None:
        import base64

        kubeconfig = (
            "apiVersion: v1\nkind: Config\n"
            "current-context: default\n"
            "contexts:\n- name: default\n  context:\n    cluster: c1\n"
            "clusters: []\nusers: []\n"
        )
        secret = MagicMock()
        secret.data = {"kubeconfig": base64.b64encode(kubeconfig.encode()).decode()}
        secret.string_data = None
        mock_core.read_namespaced_secret.return_value = secret


class TestGPUDiscoveryClient(_DiscoveryTestHelpers):
    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_discover_single_gpu_type(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        self._mock_kubeconfig_secret(mock_core)

        nodes = MagicMock()
        nodes.items = [
            self._make_node("node-1", nvidia_gpus=4),
            self._make_node("node-2", nvidia_gpus=4),
        ]

        mock_api_client = MagicMock()
        mock_k8s_config.new_client_from_config_dict.return_value = mock_api_client
        mock_k8s_client.CoreV1Api.return_value.list_node.return_value = nodes

        result = discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

        assert result.total_gpus == 8
        assert len(result.gpus) == 1
        assert result.gpus[0].short_name == "a100"
        assert result.gpus[0].count == 8
        assert result.gpus[0].node_count == 2
        assert result.gpus[0].vendor == "nvidia"
        mock_api_client.close.assert_called_once()

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_discover_mixed_gpu_types(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        self._mock_kubeconfig_secret(mock_core)

        nodes = MagicMock()
        nodes.items = [
            self._make_node("node-1", nvidia_gpus=4, nvidia_model="NVIDIA-A100-SXM4-80GB"),
            self._make_node("node-2", nvidia_gpus=8, nvidia_model="NVIDIA-H200-SXM-141GB"),
        ]

        mock_api_client = MagicMock()
        mock_k8s_config.new_client_from_config_dict.return_value = mock_api_client
        mock_k8s_client.CoreV1Api.return_value.list_node.return_value = nodes

        result = discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

        assert result.total_gpus == 12
        assert len(result.gpus) == 2
        short_names = {g.short_name for g in result.gpus}
        assert short_names == {"a100", "h200"}

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_discover_no_gpus(self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock) -> None:
        discovery, mock_core = self._make_client()
        self._mock_kubeconfig_secret(mock_core)

        nodes = MagicMock()
        nodes.items = [self._make_node("node-1")]

        mock_api_client = MagicMock()
        mock_k8s_config.new_client_from_config_dict.return_value = mock_api_client
        mock_k8s_client.CoreV1Api.return_value.list_node.return_value = nodes

        result = discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

        assert result.total_gpus == 0
        assert len(result.gpus) == 0

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_discover_amd_gpus(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        self._mock_kubeconfig_secret(mock_core)

        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", amd_gpus=8)]

        mock_api_client = MagicMock()
        mock_k8s_config.new_client_from_config_dict.return_value = mock_api_client
        mock_k8s_client.CoreV1Api.return_value.list_node.return_value = nodes

        result = discovery.discover_gpus("mi300x", "kubeconfig-mi300x", "psap-secrets")

        assert result.total_gpus == 8
        assert result.gpus[0].vendor == "amd"
        assert result.gpus[0].short_name == "mi300x"

    def test_kubeconfig_secret_not_found(self) -> None:
        discovery, mock_core = self._make_client()
        mock_core.read_namespaced_secret.side_effect = ApiException(status=404)

        with pytest.raises(GPUDiscoveryError, match="not found"):
            discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

    def test_management_api_error_wrapped(self) -> None:
        discovery, mock_core = self._make_client()
        mock_core.read_namespaced_secret.side_effect = ApiException(status=403)

        with pytest.raises(GPUDiscoveryError, match="Failed to read kubeconfig secret"):
            discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

    def test_kubeconfig_secret_no_key(self) -> None:
        discovery, mock_core = self._make_client()
        secret = MagicMock()
        secret.data = {"other-key": "value"}
        secret.string_data = None
        mock_core.read_namespaced_secret.return_value = secret

        with pytest.raises(GPUDiscoveryError, match="no 'kubeconfig' key"):
            discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_double_base64_kubeconfig(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        import base64 as b64

        discovery, mock_core = self._make_client()

        kubeconfig = (
            "apiVersion: v1\nkind: Config\n"
            "current-context: default\n"
            "contexts:\n- name: default\n  context:\n    cluster: c1\n"
            "clusters: []\nusers: []\n"
        )
        double_encoded = b64.b64encode(b64.b64encode(kubeconfig.encode())).decode()

        secret = MagicMock()
        secret.data = {"kubeconfig": double_encoded}
        secret.string_data = None
        mock_core.read_namespaced_secret.return_value = secret

        nodes = MagicMock()
        nodes.items = []
        mock_api_client = MagicMock()
        mock_k8s_config.new_client_from_config_dict.return_value = mock_api_client
        mock_k8s_client.CoreV1Api.return_value.list_node.return_value = nodes

        result = discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")
        assert result.total_gpus == 0

    def test_invalid_kubeconfig_yaml(self) -> None:
        import base64 as b64

        discovery, mock_core = self._make_client()
        secret = MagicMock()
        secret.data = {"kubeconfig": b64.b64encode(b"just-a-plain-string").decode()}
        secret.string_data = None
        mock_core.read_namespaced_secret.return_value = secret

        with pytest.raises(GPUDiscoveryError, match="not valid YAML"):
            discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

    def test_invalid_base64_raises_discovery_error(self) -> None:
        discovery, mock_core = self._make_client()
        secret = MagicMock()
        secret.data = {"kubeconfig": "not-valid-base64!!!"}
        secret.string_data = None
        mock_core.read_namespaced_secret.return_value = secret

        with pytest.raises(GPUDiscoveryError, match="invalid base64"):
            discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

    def test_unparseable_yaml_raises_discovery_error(self) -> None:
        import base64 as b64

        discovery, mock_core = self._make_client()
        secret = MagicMock()
        secret.data = {"kubeconfig": b64.b64encode(b"{{invalid: yaml: [").decode()}
        secret.string_data = None
        mock_core.read_namespaced_secret.return_value = secret

        with pytest.raises(GPUDiscoveryError, match="not valid YAML"):
            discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_target_cluster_unreachable(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        self._mock_kubeconfig_secret(mock_core)

        mock_api_client = MagicMock()
        mock_k8s_config.new_client_from_config_dict.return_value = mock_api_client
        mock_k8s_client.CoreV1Api.return_value.list_node.side_effect = Exception(
            "Connection refused"
        )

        with pytest.raises(GPUDiscoveryError, match="Failed to connect"):
            discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

        mock_api_client.close.assert_called_once()

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_auto_sets_current_context(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()

        kubeconfig_yaml = (
            "apiVersion: v1\nkind: Config\n"
            "contexts:\n- name: my-ctx\n  context:\n    cluster: c1\n"
            "clusters: []\nusers: []\n"
        )
        import base64

        secret = MagicMock()
        secret.data = {"kubeconfig": base64.b64encode(kubeconfig_yaml.encode()).decode()}
        secret.string_data = None
        mock_core.read_namespaced_secret.return_value = secret

        nodes = MagicMock()
        nodes.items = []
        mock_api_client = MagicMock()
        mock_k8s_config.new_client_from_config_dict.return_value = mock_api_client
        mock_k8s_client.CoreV1Api.return_value.list_node.return_value = nodes

        discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

        config_dict = mock_k8s_config.new_client_from_config_dict.call_args[0][0]
        assert config_dict["current-context"] == "my-ctx"

    @patch("hearth.core.gpu_discovery.k8s_config")
    def test_invalid_kubeconfig_structure(self, mock_k8s_config: MagicMock) -> None:
        discovery, mock_core = self._make_client()
        self._mock_kubeconfig_secret(mock_core)

        mock_k8s_config.new_client_from_config_dict.side_effect = Exception(
            "kubeconfig missing clusters"
        )

        with pytest.raises(GPUDiscoveryError, match="Invalid kubeconfig"):
            discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

    def test_no_contexts_raises(self) -> None:
        discovery, mock_core = self._make_client()

        import base64

        kubeconfig_yaml = "apiVersion: v1\nkind: Config\ncontexts: []\nclusters: []\nusers: []\n"
        secret = MagicMock()
        secret.data = {"kubeconfig": base64.b64encode(kubeconfig_yaml.encode()).decode()}
        secret.string_data = None
        mock_core.read_namespaced_secret.return_value = secret

        with pytest.raises(GPUDiscoveryError, match="has no contexts"):
            discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_sets_request_timeout(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        self._mock_kubeconfig_secret(mock_core)

        nodes = MagicMock()
        nodes.items = []
        mock_api_client = MagicMock()
        mock_k8s_config.new_client_from_config_dict.return_value = mock_api_client
        mock_k8s_client.CoreV1Api.return_value.list_node.return_value = nodes

        discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

        assert mock_api_client.configuration.request_timeout is not None


class TestNodeFreeGPUCapacity(_DiscoveryTestHelpers):
    @staticmethod
    def _make_pod(
        node_name: str | None,
        gpus: int = 0,
        init_gpus: int = 0,
        sidecar_gpus: int = 0,
        sidecar_first: bool = False,
        phase: str = "Running",
    ) -> SimpleNamespace:
        def container(count: int, restart_policy: str | None = None) -> SimpleNamespace:
            resources = MagicMock()
            resources.requests = {"nvidia.com/gpu": str(count)} if count else {}
            return SimpleNamespace(resources=resources, restart_policy=restart_policy)

        init_containers = []
        if sidecar_first and sidecar_gpus:
            init_containers.append(container(sidecar_gpus, restart_policy="Always"))
        if init_gpus:
            init_containers.append(container(init_gpus))
        if sidecar_gpus and not sidecar_first:
            init_containers.append(container(sidecar_gpus, restart_policy="Always"))

        return SimpleNamespace(
            spec=SimpleNamespace(
                node_name=node_name,
                containers=[container(gpus)],
                init_containers=init_containers,
            ),
            status=SimpleNamespace(phase=phase),
        )

    def _run(
        self,
        discovery: GPUDiscoveryClient,
        mock_core: MagicMock,
        mock_k8s_client: MagicMock,
        mock_k8s_config: MagicMock,
        nodes: MagicMock,
        pod_items: list | None,
    ) -> DiscoveryResult:
        self._mock_kubeconfig_secret(mock_core)
        mock_api_client = MagicMock()
        mock_k8s_config.new_client_from_config_dict.return_value = mock_api_client
        core = mock_k8s_client.CoreV1Api.return_value
        core.list_node.return_value = nodes
        if pod_items is None:
            core.list_pod_for_all_namespaces.side_effect = Exception("pods forbidden")
        else:
            pod_list = MagicMock()
            pod_list.items = pod_items
            core.list_pod_for_all_namespaces.return_value = pod_list
        return discovery.discover_gpus("cluster-1", "kubeconfig-cluster-1", "psap-secrets")

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_free_gpus_are_per_node(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [
            self._make_node("node-1", nvidia_gpus=4),
            self._make_node("node-2", nvidia_gpus=4),
        ]
        pods = [self._make_pod("node-1", gpus=4)]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, pods)

        assert [(n.name, n.total, n.free) for n in result.nodes] == [
            ("node-1", 4, 0),
            ("node-2", 4, 4),
        ]

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_pods_add_up_within_a_node(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", nvidia_gpus=8)]
        pods = [
            self._make_pod("node-1", gpus=4),
            self._make_pod("node-1", gpus=4),
        ]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, pods)

        assert result.nodes[0].free == 0

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_terminal_and_unscheduled_pods_do_not_count(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", nvidia_gpus=8)]
        pods = [
            self._make_pod("node-1", gpus=4, phase="Succeeded"),
            self._make_pod("node-1", gpus=4, phase="Failed"),
            self._make_pod(None, gpus=8, phase="Pending"),
        ]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, pods)

        assert result.nodes[0].free == 8

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_init_container_requests_count(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", nvidia_gpus=8)]
        pods = [self._make_pod("node-1", gpus=2, init_gpus=6)]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, pods)

        # The pod holds the larger of its init and app totals, not their sum.
        assert result.nodes[0].free == 2

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_restartable_init_sidecar_adds_to_app_requests(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", nvidia_gpus=8)]
        pods = [self._make_pod("node-1", gpus=4, sidecar_gpus=4)]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, pods)

        # A restartable init container keeps running next to the app ones.
        assert result.nodes[0].free == 0

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_sidecar_after_regular_init_does_not_add_to_it(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", nvidia_gpus=16)]
        pods = [self._make_pod("node-1", init_gpus=8, sidecar_gpus=8)]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, pods)

        # The sidecar starts after the init container, so init sees only 8.
        # The app phase (no app GPU requests) still holds the sidecar's 8.
        assert result.nodes[0].free == 8

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_sidecar_before_regular_init_adds_to_it(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", nvidia_gpus=16)]
        pods = [self._make_pod("node-1", init_gpus=8, sidecar_gpus=8, sidecar_first=True)]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, pods)

        # The sidecar is already running when the init container starts.
        assert result.nodes[0].free == 0

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_unlabeled_node_uses_vendor_as_gpu_type(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        node = self._make_node("node-1", nvidia_gpus=8)
        node.metadata.labels = {}
        nodes = MagicMock()
        nodes.items = [node]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, [])

        # Discovery reports this node as "nvidia"; the free block must agree.
        assert result.gpus[0].short_name == "nvidia"
        assert [(n.short_name, n.free) for n in result.nodes] == [("nvidia", 8)]

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_mixed_vendor_node_publishes_one_block_per_type(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", nvidia_gpus=4, amd_gpus=4)]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, [])

        assert [(n.short_name, n.total, n.free) for n in result.nodes] == [
            ("a100", 4, 4),
            ("mi300x", 4, 4),
        ]

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_missing_ready_condition_is_not_ready(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        node = self._make_node("node-1", nvidia_gpus=8)
        node.status.conditions = []
        nodes = MagicMock()
        nodes.items = [node]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, [])

        assert result.nodes[0].total == 8
        assert result.nodes[0].free == 0

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_not_ready_node_has_zero_free(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", nvidia_gpus=8, ready=False)]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, [])

        assert result.nodes[0].total == 8
        assert result.nodes[0].free == 0

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_cordoned_node_has_zero_free(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", nvidia_gpus=8, unschedulable=True)]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, [])

        assert result.nodes[0].free == 0

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_free_never_goes_negative(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", nvidia_gpus=4)]
        pods = [self._make_pod("node-1", gpus=8)]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, pods)

        assert result.nodes[0].free == 0

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_cpu_only_nodes_are_not_published(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [
            self._make_node("cpu-node"),
            self._make_node("node-1", nvidia_gpus=4),
        ]

        result = self._run(discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, [])

        assert [n.name for n in result.nodes] == ["node-1"]

    @patch("hearth.core.gpu_discovery.k8s_config")
    @patch("hearth.core.gpu_discovery.client")
    def test_pod_listing_failure_leaves_capacity_unknown(
        self, mock_k8s_client: MagicMock, mock_k8s_config: MagicMock
    ) -> None:
        discovery, mock_core = self._make_client()
        nodes = MagicMock()
        nodes.items = [self._make_node("node-1", nvidia_gpus=8)]

        result = self._run(
            discovery, mock_core, mock_k8s_client, mock_k8s_config, nodes, pod_items=None
        )

        # GPU totals still report; free blocks are unknown rather than zero.
        assert result.total_gpus == 8
        assert result.nodes == ()


class TestDiscoveredGPU:
    def test_frozen(self) -> None:
        gpu = DiscoveredGPU(vendor="nvidia", model="A100", short_name="a100", count=8, node_count=2)
        with pytest.raises(AttributeError):
            gpu.count = 16

    def test_fields(self) -> None:
        gpu = DiscoveredGPU(
            vendor="amd", model="MI300X", short_name="mi300x", count=4, node_count=1
        )
        assert gpu.vendor == "amd"
        assert gpu.short_name == "mi300x"
        assert gpu.count == 4


class TestDiscoveryResult:
    def test_frozen(self) -> None:
        result = DiscoveryResult(gpus=(), total_gpus=0, timestamp="2026-04-29T00:00:00Z")
        with pytest.raises(AttributeError):
            result.total_gpus = 10
