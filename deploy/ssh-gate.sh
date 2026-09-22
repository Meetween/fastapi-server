#!/usr/bin/env bash
# Forced command for the deployment key. Installed as /usr/local/bin/meetween-deploy-gate and
# referenced from authorized_keys:
#
#   command="/usr/local/bin/meetween-deploy-gate",no-pty,no-port-forwarding,no-agent-forwarding,no-X11-forwarding ssh-ed25519 AAAA... meetween-ci-deploy
#
# The key can run "deploy <ref>" and "status", nothing else. It lives outside the checkout so
# a broken ref cannot break the gate.
set -euo pipefail

# A forced command runs without the login shell, so uv in ~/.local/bin is not on PATH.
export PATH="$HOME/.local/bin:$PATH"
CHECKOUT=/mnt/vdc/meetween/fastapi-server

case "${SSH_ORIGINAL_COMMAND:-}" in
  status)
    exec systemctl status meetween-fastapi --no-pager
    ;;
  deploy\ *)
    ref="${SSH_ORIGINAL_COMMAND#deploy }"
    if [[ ! "$ref" =~ ^[A-Za-z0-9._/-]+$ ]]; then
      echo "refused: bad ref" >&2
      exit 2
    fi
    exec "$CHECKOUT/deploy/deploy.sh" "$ref"
    ;;
  *)
    echo "refused: ${SSH_ORIGINAL_COMMAND:-<none>}" >&2
    exit 2
    ;;
esac
