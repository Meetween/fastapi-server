#!/usr/bin/env bash
# Forced command for the deployment keys. Installed as /usr/local/bin/meetween-deploy-gate.
# One key per environment, each pinned to it in ~ubuntu/.ssh/authorized_keys, one line each:
#
#   restrict,command="/usr/local/bin/meetween-deploy-gate dev" ssh-ed25519 AAAA... meetween-ci-dev
#   restrict,command="/usr/local/bin/meetween-deploy-gate prod" ssh-ed25519 AAAA... meetween-ci-prod
#
# restrict turns off the pty and every forwarding. The environment comes from the key, never
# from the client: the client sends only "status" or "deploy <tag>", in SSH_ORIGINAL_COMMAND.
# The dev key therefore cannot touch prod, and the prod key takes only release tags.
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

env="${1:-}"
case "$env" in
  dev | prod) ;;
  *)
    echo "refused: unknown environment '$env'" >&2
    exit 2
    ;;
esac

# Same rules as meetween-deploy, checked here too so a refusal costs no docker pull.
valid_tag() {
  case "$env" in
    prod) [[ "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] ;;
    dev) [[ "$1" =~ ^(v[0-9]+\.[0-9]+\.[0-9]+|sha-[0-9a-f]{40}|main)$ ]] ;;
  esac
}

cmd="${SSH_ORIGINAL_COMMAND:-}"
if [ "$cmd" = status ]; then
  systemctl status "meetween-fastapi@$env" --no-pager -n 0 || true
  grep '^IMAGE=' "/mnt/vdc/meetween/$env/image.env" 2>/dev/null || echo "IMAGE not set"
  port=$(sudo sed -n 's/^PORT=//p' "/etc/meetween/$env.env" | tail -n 1)
  # Exit code 0 only when /health answers, so a caller can use status as a check.
  if curl -sf --max-time 5 "http://127.0.0.1:$port/health"; then
    echo
    exit 0
  fi
  echo "no answer from http://127.0.0.1:$port/health" >&2
  exit 1
elif [[ "$cmd" =~ ^deploy\ ([^[:space:]]+)$ ]]; then
  tag="${BASH_REMATCH[1]}"
  if ! valid_tag "$tag"; then
    echo "refused: bad tag for $env: $tag" >&2
    exit 2
  fi
  exec /usr/local/bin/meetween-deploy "$env" "$tag"
fi
echo "refused: ${cmd:-<none>}" >&2
exit 2
