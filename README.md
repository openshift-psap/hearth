# hearth
Kubernetes operator that manages GPU cluster lifecycle for Fournos. auto-discovers clusters via labeled kubeconfig secrets, validates connectivity, discovers GPU hardware, and dynamically manages Kueue ResourceFlavors and ClusterQueue quotas.
