# Deployment

The service runs on the shared VM as the systemd unit `meetween-fastapi`, on port 8001, from
the release checkout `/mnt/vdc/meetween/fastapi-server`. Jigasi and the Jitsi web frontend
reach it at `host.docker.internal:8001`.

## How a release deploys

1. A pull request is squash merged into `main`.
2. release-please updates the open `chore(main): release x.y.z` pull request.
3. Merging that pull request creates the tag `vX.Y.Z` and the GitHub Release.
4. `release: draft or cut` dispatches `deploy: vm` at the tag.
5. `deploy/deploy.sh` runs on the VM: fetch, checkout, `uv sync --locked --no-dev`, restart,
   poll `/health`, compare the reported version with `pyproject.toml`, roll back on mismatch.

## Deploy a branch or roll back

```sh
gh workflow run deploy.yml -f ref=feat/my-branch
gh workflow run deploy.yml -f ref=v0.2.0
```

On the VM, the same without GitHub:

```sh
/mnt/vdc/meetween/fastapi-server/deploy/deploy.sh v0.2.0
```

## Logs and status

```sh
journalctl -u meetween-fastapi -f
systemctl status meetween-fastapi
curl -s localhost:8001/health
```

## Files outside git

Copy them again when their source in `deploy/` changes.

| On the VM | Source |
| --- | --- |
| `/etc/systemd/system/meetween-fastapi.service` | `deploy/meetween-fastapi.service` |
| `/etc/meetween/fastapi.env` | `deploy/fastapi.env.example`, with the real values |
| `/usr/local/bin/meetween-deploy-gate` | `deploy/ssh-gate.sh` |

## Bootstrap, once

As `ubuntu` on the VM.

```sh
# 1. Room on the root disk: uv needs its cache on /mnt/vdc, but apt and systemd need /.
sudo journalctl --vacuum-size=200M && sudo apt-get clean && df -h /

# 2. Directories.
sudo mkdir -p /mnt/vdc/meetween /etc/meetween && sudo chown ubuntu:ubuntu /mnt/vdc/meetween

# 3. Read only deploy key for git fetch. Add the public half on GitHub:
#    gh repo deploy-key add ~/.ssh/meetween-deploy-key.pub -R Meetween/fastapi-server -t speechlmm-readonly
ssh-keygen -t ed25519 -f ~/.ssh/meetween-deploy-key -N "" -C "speechlmm deploy key"

# 4. Release checkout, detached at the ref to run.
GIT_SSH_COMMAND="ssh -i ~/.ssh/meetween-deploy-key -o IdentitiesOnly=yes" \
  git clone git@github.com:Meetween/fastapi-server.git /mnt/vdc/meetween/fastapi-server
cd /mnt/vdc/meetween/fastapi-server
git config core.sshCommand "ssh -i ~/.ssh/meetween-deploy-key -o IdentitiesOnly=yes"
git checkout --detach v0.1.0

# 5. Dependencies. uv picks /usr/bin/python3.12 from .python-version.
UV_CACHE_DIR=/mnt/vdc/meetween/uv-cache uv sync --locked --no-dev

# 6. Configuration, unit and gate.
sudo cp deploy/fastapi.env.example /etc/meetween/fastapi.env && sudo chmod 600 /etc/meetween/fastapi.env
sudo vi /etc/meetween/fastapi.env
sudo cp deploy/meetween-fastapi.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable meetween-fastapi
sudo install -m 755 deploy/ssh-gate.sh /usr/local/bin/meetween-deploy-gate
```

On a laptop, the key the workflow uses:

```sh
ssh-keygen -t ed25519 -f meetween-ci-deploy -N "" -C meetween-ci-deploy
# Append to ~ubuntu/.ssh/authorized_keys on the VM, on one line:
#   command="/usr/local/bin/meetween-deploy-gate",no-pty,no-port-forwarding,no-agent-forwarding,no-X11-forwarding <contents of meetween-ci-deploy.pub>
gh secret set VM_SSH_KEY -R Meetween/fastapi-server < meetween-ci-deploy
gh variable set VM_HOST -R Meetween/fastapi-server -b "<vm-host>"
gh variable set VM_USER -R Meetween/fastapi-server -b ubuntu
gh variable set VM_KNOWN_HOSTS -R Meetween/fastapi-server -b "$(ssh-keyscan -t ed25519 "<vm-host>" 2>/dev/null)"
rm meetween-ci-deploy meetween-ci-deploy.pub

# The gate answers status and refuses everything else.
ssh -i meetween-ci-deploy ubuntu@"<vm-host>" status
ssh -i meetween-ci-deploy ubuntu@"<vm-host>" ls
```

## Cutover from a process started by hand

Only one process can hold port 8001. Stop the old one with SIGINT so uvicorn runs its
shutdown hooks, then start the unit and check the version.

```sh
ss -ltnp | grep :8001
kill -INT <pid>
sudo systemctl start meetween-fastapi
curl -s localhost:8001/health
```

Rollback: `sudo systemctl stop meetween-fastapi` and start the old process the way it was
started before.
