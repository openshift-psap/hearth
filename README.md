# Hearth

Kubernetes operator that manages GPU cluster lifecycle for Fournos. Auto-discovers clusters via labeled kubeconfig secrets, validates connectivity, discovers GPU hardware, and dynamically manages Kueue ResourceFlavors and ClusterQueue quotas.

> **Naming note:** Hearth was extracted from the `fournos-cluster` controller. The CRD is still `FournosCluster` (`fournos.dev/v1`), labels use `fournos.dev/*`, and Kueue resources use `fournos/*`. Only the operator package, namespace, and deployment were renamed.

## Prerequisites

- `oc` CLI authenticated to the psap-automation management cluster
- `podman` for building container images
- Python 3.12+ for local development
- Access to `quay.io/rh_perfscale/hearth` image repository

## Architecture

Hearth runs as a single-replica Deployment in the `hearth` namespace on the psap-automation management cluster.

| Namespace | What it watches |
|-----------|-----------------|
| `psap-secrets` | Secrets labeled `fournos.dev/cluster-kubeconfig=true` |
| `hearth` | FournosCluster custom resources |

When a labeled kubeconfig secret is detected, the controller:

1. Creates a `FournosCluster` CR in the `hearth` namespace
2. Validates the kubeconfig by connecting to the target cluster
3. Discovers GPUs on the target cluster via node labels
4. Creates a Kueue `ResourceFlavor` and adds it to the `ClusterQueue`

### CD Flow

```
PR merged to main
  -> GitHub Actions builds + pushes quay.io/rh_perfscale/hearth:latest
    -> OpenShift ImageStream detects new image (~15 min poll)
      -> image trigger rolls out new pod
        -> ArgoCD ignores image field (ignoreDifferences)
```

## Deployment

Hearth is deployed via ArgoCD. The ArgoCD Application manifest lives in this repo at `argocd/app-hearth.yaml`.

### First-time setup

```bash
# Apply the CRD
oc apply -f manifests/crd.yaml

# Create the ArgoCD Application (one-time, ArgoCD manages everything after this)
oc apply -f argocd/app-hearth.yaml

# Set up image pull secret for quay.io (private repo)
oc create secret docker-registry quay-pull-secret \
  --docker-server=quay.io \
  --docker-username='rh_perfscale+psap_automation' \
  --docker-password='<ROBOT_TOKEN>' \
  -n hearth
oc secrets link hearth quay-pull-secret --for=pull -n hearth
```

### Verify

```bash
# Pod running
oc get pods -n hearth -l app=hearth

# Logs (filter out healthz noise)
oc logs -n hearth -l app=hearth | grep -v healthz | head -20

# RBAC
oc auth can-i --as=system:serviceaccount:hearth:hearth list fournosclusters.fournos.dev

# Discovered clusters
oc get fournoscluster -n hearth
```

## Testing from a Branch

Build with a branch-specific tag (never overwrite `:latest`):

```bash
# Build and push
podman build -t quay.io/rh_perfscale/hearth:my-branch -f Containerfile .
podman push quay.io/rh_perfscale/hearth:my-branch

# Deploy branch image
oc set image deployment/hearth hearth=quay.io/rh_perfscale/hearth:my-branch -n hearth

# Watch logs
oc logs -n hearth -l app=hearth -f | grep -v healthz

# Revert to production
oc rollout undo deployment/hearth -n hearth
```

## Cluster Onboarding

### Step 1: Create kubeconfig secret

```bash
oc create secret generic kubeconfig-<cluster-name> \
  --from-file=kubeconfig=./<cluster-name>-kubeconfig \
  -n psap-secrets
```

### Step 2: Label the secret

```bash
oc label secret kubeconfig-<cluster-name> fournos.dev/cluster-kubeconfig=true -n psap-secrets
```

### Step 3: Verify auto-discovery

Hearth detects the label and creates a FournosCluster CR:

```bash
oc get fournoscluster -n hearth
# NAME           OWNER   GPUS          KUBECONFIG   LOCKED   AGE
# cluster-name                         Valid         false    10s
```

### Step 4: Verify GPU discovery

GPU discovery runs automatically (default: every 5 minutes):

```bash
oc get fournoscluster <cluster-name> -n hearth -o jsonpath='{.status.gpuSummary}'
# Example: "2x NVIDIA-L40S"

# Detailed GPU info
oc get fournoscluster <cluster-name> -n hearth -o jsonpath='{.spec.hardware.gpus}' | python3 -m json.tool
```

### Step 5: Verify Kueue resources

```bash
# ResourceFlavor created
oc get resourceflavor <cluster-name>

# ClusterQueue updated with new flavor
oc get clusterqueue fournos-queue -o jsonpath='{.spec.resourceGroups[0].flavors}' | python3 -m json.tool
```

## Cluster Offboarding (Manual)

> Automated cleanup via `on.delete` handler is planned for a future iteration.

### Step 1: Delete the FournosCluster CR

```bash
oc delete fournoscluster <cluster-name> -n hearth
```

### Step 2: Delete the ResourceFlavor

```bash
oc delete resourceflavor <cluster-name>
```

### Step 3: Remove flavor from ClusterQueue

```bash
# Get current CQ, remove the flavor entry, and patch
oc get clusterqueue fournos-queue -o json \
  | python3 -c "
import json, sys
cq = json.load(sys.stdin)
flavors = cq['spec']['resourceGroups'][0]['flavors']
flavors[:] = [f for f in flavors if f['name'] != '<cluster-name>']
print(json.dumps({'spec': {'resourceGroups': cq['spec']['resourceGroups']}}))" \
  | oc patch clusterqueue fournos-queue --type=merge -p "$(cat -)"
```

### Step 4: Optionally remove the kubeconfig secret

```bash
oc delete secret kubeconfig-<cluster-name> -n psap-secrets
```

## Cluster Locking

Locking a cluster prevents new Kueue workloads from being scheduled on it by creating a sentinel FournosJob that consumes all `fournos/cluster-slot` quota for that cluster's flavor.

### Lock a cluster

```bash
oc patch fournoscluster <cluster-name> -n hearth \
  --type=merge -p '{"spec":{"owner":"your-name"}}'
```

### Verify the lock

```bash
# Status shows locked
oc get fournoscluster <cluster-name> -n hearth
# NAME           OWNER       GPUS          KUBECONFIG   LOCKED   AGE
# cluster-name   your-name   2x NVIDIA     Valid        true     5m

# Sentinel FournosJob exists (in execution namespace)
oc get fournosjobs -n psap-automation
# NAME                          OWNER       PHASE      CLUSTER        AGE
# cluster-lock-<cluster-name>   your-name   Admitted   cluster-name   10s

# Sentinel's Kueue Workload is admitted, consuming all cluster-slots
oc get workloads -n psap-automation
```

### Lock with TTL (auto-expiry)

```bash
oc patch fournoscluster <cluster-name> -n hearth \
  --type=merge -p '{"spec":{"owner":"your-name","ttl":"30m"}}'

# Check expiry time
oc get fournoscluster <cluster-name> -n hearth -o jsonpath='{.status.lockExpiresAt}'
```

### Unlock a cluster

```bash
oc patch fournoscluster <cluster-name> -n hearth \
  --type=merge -p '{"spec":{"owner":""}}'

# Verify: locked=false, sentinel FournosJob deleted
oc get fournoscluster <cluster-name> -n hearth -o jsonpath='{.status.locked}'
oc get fournosjobs -n psap-automation
```

## Job Queueing (Lock Verification)

While a cluster is locked, new workloads targeting it stay pending:

### Create a test workload

```bash
cat <<'EOF' | oc create -n psap-automation -f -
apiVersion: kueue.x-k8s.io/v1beta2
kind: Workload
metadata:
  name: test-blocked-job
  labels:
    kueue.x-k8s.io/queue-name: fournos-queue
spec:
  active: true
  queueName: fournos-queue
  podSets:
    - name: launcher
      count: 1
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: placeholder
              image: registry.k8s.io/pause:3.9
              resources:
                requests:
                  fournos/cluster-slot: "1"
          nodeSelector:
            fournos.dev/cluster: <cluster-name>
EOF
```

### Verify it is blocked

```bash
oc get workloads -n psap-automation
# test-blocked-job should have no RESERVED IN or ADMITTED value

oc get workload test-blocked-job -n psap-automation -o jsonpath='{.status.conditions[*].message}'
# "insufficient unused quota for fournos/cluster-slot in flavor <cluster-name>"
```

### Unlock and verify scheduling

```bash
# Unlock the cluster
oc patch fournoscluster <cluster-name> -n hearth \
  --type=merge -p '{"spec":{"owner":""}}'

# Workload should now be admitted
oc get workloads -n psap-automation
# test-blocked-job   fournos-queue   fournos-queue   True   30s

# Clean up
oc delete workload test-blocked-job -n psap-automation
```

## Configuration

Environment variables (set in `deploy/deployment.yaml`, prefix `HEARTH_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HEARTH_NAMESPACE` | `hearth` | Namespace for FournosCluster CRs |
| `HEARTH_SECRETS_NAMESPACE` | `psap-secrets` | Namespace containing kubeconfig secrets |
| `HEARTH_EXECUTION_NAMESPACE` | `psap-automation` | Namespace where FournosJobs run (sentinel jobs created here) |
| `HEARTH_RECONCILE_INTERVAL_SEC` | `30` | Timer reconciliation interval (seconds) |
| `HEARTH_GPU_DISCOVERY_DEFAULT_INTERVAL_SEC` | `300` | GPU discovery interval (seconds) |
| `HEARTH_GPU_DISCOVERY_TIMEOUT_SEC` | `10` | Target cluster API timeout (seconds) |
| `HEARTH_LOG_LEVEL` | `INFO` | Logging level |

## Troubleshooting

### Where to look

```bash
# Controller logs (filter healthz noise)
oc logs -n hearth -l app=hearth | grep -v healthz | tail -50

# Follow logs in real time
oc logs -n hearth -l app=hearth -f | grep -v healthz
```

### Pod in CrashLoopBackOff

Check for RBAC errors in logs. Common causes:
- `cannot list resource` errors: ClusterRole missing permissions. Re-apply: `oc apply -f deploy/clusterrole.yaml`
- `Forbidden` on secrets: controller needs `get`, `list`, `watch`, `patch` on secrets in both `hearth` and `psap-secrets` namespaces

### No FournosCluster CR created after labeling a secret

1. Verify the label: `oc get secret <name> -n psap-secrets --show-labels | grep cluster-kubeconfig`
2. Verify the secret has a `kubeconfig` key: `oc get secret <name> -n psap-secrets -o jsonpath='{.data}' | python3 -m json.tool`
3. Check controller logs for errors

### GPU discovery shows 0 GPUs or generic model

- Target cluster may not have GPU Feature Discovery (GFD) installed. Without GFD, `nvidia.com/gpu.product` labels are missing on nodes.
- GPUs are still detected via `nvidia.com/gpu` allocatable resources but show as generic `NVIDIA` instead of specific model.
- Check discovery errors: `oc get fournoscluster <name> -n hearth -o jsonpath='{.spec.hardware.lastError}'`

### Sentinel FournosJob stuck in Pending

The sentinel needs Kueue to admit its Workload. Check if another workload is consuming cluster-slots:

```bash
oc get workloads -n psap-automation
```

### Kueue changes being reverted

If ResourceFlavor or ClusterQueue changes disappear after a few minutes, ArgoCD's `selfHeal` may be reverting them. Verify that `ignoreDifferences` is configured on the `fournos-cluster` ArgoCD Application for `ClusterQueue.spec.resourceGroups` and `ResourceFlavor.spec`. See [fournos-gitops PR #10](https://github.com/openshift-psap/fournos-gitops/pull/10).

## Local Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=hearth --cov-report=term-missing

# Lint
ruff check hearth/ tests/
ruff format --check hearth/ tests/
```
