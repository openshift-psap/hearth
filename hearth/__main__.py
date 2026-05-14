import kopf

from hearth import operator  # noqa: F401 — registers kopf handlers
from hearth.settings import settings

kopf.run(
    namespaces=[settings.namespace, settings.secrets_namespace],
    liveness_endpoint="http://0.0.0.0:8080/healthz",
)
