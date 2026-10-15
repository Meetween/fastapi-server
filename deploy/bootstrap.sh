#!/usr/bin/env bash
# Prepares one environment on the VM for container deploys. Run by hand as ubuntu from a copy
# of deploy/, once per environment and again whenever a file here changes:
#
#   ./bootstrap.sh <dev|prod>
#
# Idempotent. It never overwrites /etc/meetween/<env>.env and never starts the unit: the first
# meetween-deploy writes IMAGE into image.env and starts it. Before that the unit has nothing to
# run. It touches only /mnt/vdc/meetween/<env>, /etc/meetween/<env>.env, the unit template and
# the two scripts in /usr/local/bin. authorized_keys is left alone; the line to add is printed.
set -euo pipefail

env="${1:-}"
case "$env" in
  dev) port=8011 db=meetween_dev ;;
  prod) port=8012 db=meetween ;;
  *)
    echo "usage: bootstrap.sh <dev|prod>" >&2
    exit 2
    ;;
esac

here=$(cd "$(dirname "$0")" && pwd)
dir="/mnt/vdc/meetween/$env"
recordings="$dir/recordings"
envfile="/etc/meetween/$env.env"

run() {
  echo "+ $*"
  "$@"
}

# uid 1000 is ubuntu on the VM and the user inside the image: meetween-deploy writes image.env
# here as ubuntu, the container writes recordings as 1000.
for d in "$dir" "$recordings"; do
  if [ ! -d "$d" ]; then
    run sudo install -d -o 1000 -g 1000 -m 0755 "$d"
  elif [ "$(stat -c %u "$d")" != 1000 ]; then
    run sudo chown 1000:1000 "$d"
  else
    echo "ok $d"
  fi
done

if [ ! -e "$dir/image.env" ]; then
  echo "+ write $dir/image.env"
  echo "# IMAGE=ghcr.io/meetween/fastapi-server@sha256:<digest> is written by meetween-deploy on the first deploy." \
    > "$dir/image.env"
  chmod 0644 "$dir/image.env"
else
  echo "ok $dir/image.env"
fi

if sudo test -e "$envfile"; then
  echo "ok $envfile, left as it is"
else
  if [ ! -d /etc/meetween ]; then
    run sudo install -d -o root -g root -m 0755 /etc/meetween
  fi
  echo "+ write $envfile from fastapi.env.example, root:root 0600, new JWT secret"
  # The secret reaches awk through its environment, which only ubuntu and root can read, not
  # through its arguments, which every user sees in ps. It is never written to a file other
  # than the root owned one.
  secret=$(openssl rand -hex 32)
  if [ "${#secret}" -ne 64 ]; then
    echo "openssl rand failed" >&2
    exit 1
  fi
  JWT_SECRET_KEY=$secret PORT=$port MONGO_DB_NAME=$db RECORDINGS_BASE_PATH=$recordings \
    awk -F= '$1 ~ /^(PORT|MONGO_DB_NAME|RECORDINGS_BASE_PATH|JWT_SECRET_KEY)$/ { print $1 "=" ENVIRON[$1]; next } { print }' \
    "$here/fastapi.env.example" \
    | sudo install -o root -g root -m 0600 /dev/stdin "$envfile"
  unset secret
  echo "  edit JITSI_MEET_BASE_URL and MODEL_PROFILE if needed: sudo vi $envfile"
fi

run sudo install -o root -g root -m 0644 "$here/meetween-fastapi@.service" /etc/systemd/system/meetween-fastapi@.service
run sudo install -o root -g root -m 0755 "$here/ssh-gate.sh" /usr/local/bin/meetween-deploy-gate
run sudo install -o root -g root -m 0755 "$here/meetween-deploy" /usr/local/bin/meetween-deploy
run sudo systemctl daemon-reload
run sudo systemctl enable "meetween-fastapi@$env"

cat <<MSG

$env is ready for its first deploy, the unit is enabled and not started:
  meetween-deploy $env <tag>

For CI, append one line to ~ubuntu/.ssh/authorized_keys with the public half of the $env key:
  restrict,command="/usr/local/bin/meetween-deploy-gate $env" <contents of meetween-ci-$env.pub>
MSG
