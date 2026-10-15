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
#
# A deploy may carry registry credentials on stdin, never in the command: line 1 a token that
# can read the package, line 2 the GitHub login it belongs to. CI sends its own GITHUB_TOKEN,
# which expires with the job. They are checked and handed to meetween-deploy on its stdin, and
# never printed. No stdin means no credentials: meetween-deploy then uses an image already on
# the VM, which is enough to roll back by hand.
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
  # read gives up after 10 s, so a client that keeps stdin open and sends nothing still deploys,
  # without credentials. A partial line at the timeout is refused rather than guessed at.
  token="" actor="" rc=0
  IFS= read -r -t 10 token || rc=$?
  if [ "$rc" -gt 128 ] && [ -n "$token" ]; then
    echo "refused: timed out reading the registry token" >&2
    exit 2
  fi
  if [ -z "$token" ]; then
    exec /usr/local/bin/meetween-deploy "$env" "$tag" </dev/null
  fi
  IFS= read -r -t 10 actor || true
  # A loose check: no whitespace, so the token cannot smuggle a second line or an argument, and
  # a length cap. The login allows the [bot] suffix of github-actions[bot].
  if [[ ! "$token" =~ ^[[:graph:]]+$ ]] || [ "${#token}" -gt 4096 ]; then
    echo "refused: malformed registry token (${#token} characters)" >&2
    exit 2
  fi
  if [[ ! "$actor" =~ ^[A-Za-z0-9][A-Za-z0-9-]{0,38}(\[bot\])?$ ]]; then
    echo "refused: malformed registry user" >&2
    exit 2
  fi
  # printf is a builtin, so the token is in no argv; the process substitution is a pipe, so it
  # is in no file either.
  exec /usr/local/bin/meetween-deploy "$env" "$tag" < <(printf '%s\n%s\n' "$token" "$actor")
fi
echo "refused: ${cmd:-<none>}" >&2
exit 2
