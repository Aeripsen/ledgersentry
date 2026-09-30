#!/usr/bin/env bash
# Schema check of every manifest CI applies: the kustomize base, the kind
# overlay, and the k6 load Job, rendered by kubectl and validated by kubeconform
# in strict mode (unknown fields are errors). The k8s workflow runs this on
# every deploy change; locally it needs kubectl, and fetches kubeconform if it
# is not on PATH.
#
#   bash scripts/k8s_schema.sh
#
# KUBERNETES_VERSION picks the schema set (kubeconform's default is "master",
# the newest schemas published at github.com/yannh/kubernetes-json-schema).
set -euo pipefail

KUBECONFORM_VERSION=${KUBECONFORM_VERSION:-v0.8.0}
KUBERNETES_VERSION=${KUBERNETES_VERSION:-master}

KC=$(command -v kubeconform || true)
if [ -z "$KC" ]; then
  tmp=$(mktemp -d)
  curl -sSfL "https://github.com/yannh/kubeconform/releases/download/$KUBECONFORM_VERSION/kubeconform-linux-amd64.tar.gz" \
    | tar -xz -C "$tmp" kubeconform
  KC="$tmp/kubeconform"
fi
echo "kubeconform $("$KC" -v), schemas for Kubernetes $KUBERNETES_VERSION"

check() { "$KC" -strict -summary -kubernetes-version "$KUBERNETES_VERSION" "$@"; }
for target in base overlays/kind; do
  echo "== deploy/k8s/$target"
  kubectl kustomize "deploy/k8s/$target" | check -
done
echo "== deploy/k8s/loadtest/job.yaml"
check deploy/k8s/loadtest/job.yaml
