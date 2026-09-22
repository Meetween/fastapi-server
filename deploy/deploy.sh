#!/usr/bin/env bash
# Deploys one git ref of this repository on the VM: fetch, checkout, install the locked
# dependencies, restart the systemd unit, poll /health, roll back if the reported version is
# not the one at the ref. Run as ubuntu from the release checkout, by hand or through the SSH
# gate. Exit code 0 only when the requested ref is running.
#
# The whole script is one function called on the last line. It checks out a new ref of the
# directory it is running from, and bash reads a script incrementally, so the body must be
# parsed in full before the checkout happens.

main() {
  set -euo pipefail

  local checkout=/mnt/vdc/meetween/fastapi-server
  local state=/mnt/vdc/meetween/previous-ref
  local unit=meetween-fastapi
  local health=http://127.0.0.1:8001/health
  local timeout=90
  local ref="${1:?usage: deploy.sh <tag|branch|commit>}"

  export PATH="$HOME/.local/bin:$PATH"
  export UV_CACHE_DIR=/mnt/vdc/meetween/uv-cache
  cd "$checkout"

  git fetch --tags --prune --quiet origin
  local target
  target=$(git rev-parse --verify --quiet "refs/tags/$ref^{commit}" \
    || git rev-parse --verify --quiet "refs/remotes/origin/$ref^{commit}" \
    || git rev-parse --verify --quiet "$ref^{commit}") \
    || { echo "unknown ref: $ref" >&2; exit 1; }

  local previous
  previous=$(git rev-parse HEAD)
  echo "$previous" > "$state"
  echo "deploying $ref ($target), previous $previous"

  if [ -n "$(git status --porcelain)" ]; then
    echo "warning: the release checkout has local changes, they are left in place" >&2
    git status --short >&2
  fi

  switch_and_restart "$target" "$unit"
  if verify "$health" "$timeout" "$unit"; then
    echo "deployed $ref ($target)"
    exit 0
  fi

  echo "verification failed, rolling back to $previous" >&2
  switch_and_restart "$previous" "$unit"
  if verify "$health" "$timeout" "$unit"; then
    echo "rolled back to $previous" >&2
  else
    echo "rollback failed, the service needs a hand" >&2
  fi
  exit 1
}

switch_and_restart() {
  local target="$1" unit="$2"
  git checkout --detach --quiet "$target"
  uv sync --locked --no-dev
  sudo systemctl restart "$unit"
}

verify() {
  local health="$1" timeout="$2" unit="$3"
  local expected body got
  expected=$(uv version --short)
  body=""
  for _ in $(seq 1 "$timeout"); do
    if body=$(curl -sf --max-time 2 "$health"); then
      break
    fi
    sleep 1
  done
  if [ -z "$body" ]; then
    echo "no answer from $health after ${timeout}s" >&2
    sudo journalctl -u "$unit" -n 50 --no-pager >&2
    return 1
  fi
  got=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version", ""))')
  if [ "$got" != "$expected" ]; then
    echo "version mismatch: /health reports '$got', pyproject.toml says '$expected'" >&2
    sudo journalctl -u "$unit" -n 50 --no-pager >&2
    return 1
  fi
  echo "$body"
}

main "$@"
