# Deployment

CI builds one container image, `ghcr.io/meetween/fastapi-server`, and the same image runs in both
environments. The shared VM hosts `dev` and `prod`. Each is a systemd unit,
`meetween-fastapi@<env>`, that runs the image with `docker run --network host`: host networking
keeps MongoDB on `localhost:27017` and Triton on `localhost:9000`, `9100`, `9110` reachable as
they always were. The image is pinned by digest, so a restart never picks up a moved tag. The
unit, the scripts and the env files are not in the image; `bootstrap.sh` installs them.

The package is private and the VM stores no registry credential. Each deploy from GitHub Actions
sends the job's own `GITHUB_TOKEN` over SSH, and the VM pulls with it once and throws it away.
The unit runs with `--pull=never`, so a restart or a reboot only needs the image on disk.

## Layout

| | dev | prod |
| --- | --- | --- |
| Port | 8011 | 8012, moves to 8001 at the cutover |
| Unit | `meetween-fastapi@dev` | `meetween-fastapi@prod` |
| Container | `meetween-fastapi-dev` | `meetween-fastapi-prod` |
| Config and secrets | `/etc/meetween/dev.env` | `/etc/meetween/prod.env` |
| Running image | `/mnt/vdc/meetween/dev/image.env` | `/mnt/vdc/meetween/prod/image.env` |
| Image before it | `/mnt/vdc/meetween/dev/previous-image` | `/mnt/vdc/meetween/prod/previous-image` |
| Recordings | `/mnt/vdc/meetween/dev/recordings` | `/mnt/vdc/meetween/prod/recordings` |
| Logs | `docker logs meetween-fastapi-dev` | `docker logs meetween-fastapi-prod` |
| Database | `meetween_dev` | `meetween` |

- The env files are `root:root 0600`: they hold the JWT secret. Everything else under
  `/mnt/vdc/meetween/<env>` is owned by uid 1000, the `ubuntu` user and the user inside the image.
- Docker keeps the logs as json-file, 50 MB by 5 files per container, on its own disk.
  `journalctl -u meetween-fastapi@<env>` shows only the unit starting, stopping and its exit
  status: the unit sends the container output nowhere else, so nothing reaches the journal on
  the root disk.
- Each environment has its own database, JWT secret and recordings tree, so a dev token is
  never valid on prod and the 02:00 retention cleanup of one cannot delete files of the other.

## How a change reaches dev

1. A pull request is squash merged into `main`.
2. `build: image` builds the image and pushes `sha-<sha>` and `main`.
3. `deploy: vm` runs when that build succeeds and deploys `sha-<sha>` to dev, but only while the
   repository variable `DEPLOY_DEV_ENABLED` is `true`. Set it to anything else to pause dev.

Pull requests only build the image, without pushing. A branch cannot be deployed: images are
published from `main` only.

## How a release reaches prod

1. release-please keeps one pull request open, `chore(main): release x.y.z`.
2. Its checks wait as "action required", because the release is raised with the default token.
   A maintainer clicks "Approve workflows to run" on it so `Test` and `Conventional Commits
   title` run.
3. Merging it creates the tag `vX.Y.Z` and the GitHub Release. The same push builds
   `sha-<sha>`, which also goes to dev.
4. `release: draft or cut` promotes `sha-<sha>` to `vX.Y.Z`. This is a second name for the same
   digest, not a rebuild, so prod runs the bytes that were built and could be tried on dev. The
   job waits up to 20 minutes for the build.
5. It then dispatches `deploy: vm` for prod. The `prod` environment holds the run until a
   maintainer approves it in the Actions UI.

## Manual deploy and rollback

Always dispatch on `main` with the tag as an input. A dispatch runs the copy of `deploy.yml` at
the ref it is dispatched on, so a dispatch at an old tag would run an old deploy procedure
against today's VM.

```sh
# prod: a release tag only. Approve the run in the prod environment.
gh workflow run deploy.yml --ref main -f environment=prod -f tag=v0.2.0

# dev: v0.2.0, main, or sha-<40 hex commit>; no tag means the head of main.
gh workflow run deploy.yml --ref main -f environment=dev -f tag=main
gh workflow run deploy.yml --ref main -f environment=dev -f tag=sha-<40 hex commit>
```

The image must exist in GHCR. `release: draft or cut` creates it for new releases. For a release
cut before the pipeline, backfill it first, again on `main`:

```sh
gh workflow run image.yml --ref main -f ref=v0.2.0
```

`ref` is a release tag or a full 40 character commit SHA. The build uses the Dockerfile stored at
that ref, so this works only for a tag that contains one. A tag without a Dockerfile (`v0.1.0` and
`v0.2.0` today) fails to build; roll back to the oldest tag that has an image.

To undo a bad change for good, revert it in a pull request titled as a `fix`, for example
`fix(summary): revert ...`. release-please bumps the patch for a `fix`, so the revert ships as the
next patch release.

## How a deploy pulls the image

`deploy: vm` runs with `permissions: contents: read, packages: read`. Its SSH step writes two
lines to the standard input of `ssh`, line 1 the job's `GITHUB_TOKEN`, line 2 `github.actor`;
the remote command stays `deploy <tag>`. Both go through `env:` and `printf`, a shell builtin,
so neither is in a command line on the runner or on the VM.

On the VM `meetween-deploy-gate` reads at most those two lines, waiting 10 seconds for each,
checks their shape (a token of 1 to 4096 characters without whitespace, a GitHub login with an
optional `[bot]` suffix) and hands them to `meetween-deploy` on its stdin. A malformed token
or login is refused with exit 2 before docker runs. `status` reads nothing.

`meetween-deploy` then logs in to `ghcr.io` in a temporary docker config, mode 700, with
`--password-stdin`, pulls, logs out and deletes the directory; an exit trap deletes it on any
failure too. Nothing reaches `~/.docker` of the deploy user. The token is valid for this
repository only and expires when the job ends.

Without credentials on stdin `meetween-deploy` uses the image for the tag if it is already on the
VM, and says it did not pull. A moving tag such as `main` may then be older than the registry's;
the digest it prints tells which build it is. If the image is not there it tries an anonymous
pull, which fails while the package is private, and exits 1 with "image not present and no
registry credentials" without changing anything.

## On the VM

```sh
meetween-deploy <dev|prod> <tag>      # dev: vX.Y.Z, main, sha-<40 hex>; prod: vX.Y.Z
meetween-deploy prod v0.2.0
```

By hand, without a token, this works for any image still on the VM: the three most recent and
whatever `image.env` and `previous-image` point to, which covers a rollback.
`docker image ls ghcr.io/meetween/fastapi-server` lists them. For any other tag dispatch
`deploy: vm` instead. Should the registry ever have to be reached by hand, pass a classic
personal access token with only `read:packages` on stdin, never as an argument, and revoke it
afterwards:

```sh
read -rs TOKEN                       # paste the token, it is not echoed
printf '%s\nmime-pi\n' "$TOKEN" | meetween-deploy dev <tag>
unset TOKEN
```

Status, logs, restart:

```sh
systemctl status meetween-fastapi@<env>
cat /mnt/vdc/meetween/<env>/image.env
curl -s http://127.0.0.1:<port>/health
docker logs -f meetween-fastapi-<env>
sudo systemctl restart meetween-fastapi@<env>
```

`meetween-deploy` does this, in order:

1. Takes the lock `/mnt/vdc/meetween/<env>/.deploy.lock`, so two deploys of one environment
   cannot race on `image.env`. A second one exits at once.
2. Pulls `ghcr.io/meetween/fastapi-server:<tag>` with the credentials from stdin, or without
   them uses the copy already on the VM (see above), and resolves its digest.
3. If that digest already runs and `/health` is right, prints "already runs" and exits 0.
4. Copies the current image to `previous-image`, writes the new digest to `image.env` and
   restarts the unit.
5. Polls `/health` for up to 90 seconds. It must report the version in the image label
   `org.opencontainers.image.version`, and the container must run that exact image. The version
   alone cannot tell two builds of one release apart.
6. On failure it prints the last 50 log lines and puts `previous-image` back, restarts and checks
   again. On a first deploy there is nothing to go back to: it restores `image.env` and stops the
   unit so it does not restart a broken image in a loop.
7. On success it prunes: the 3 most recent images stay, plus any image that an `image.env`, a
   `previous-image` or a running container refers to.

| Exit | Meaning |
| --- | --- |
| 0 | The requested tag is running |
| 1 | Not running: verification failed (rolled back, or stopped on a first deploy), rollback failed, lock held, `bootstrap.sh` not run, no `PORT` in the env file, login, pull or digest failed, image not present and no credentials |
| 2 | Refused before any change: wrong argument count, unknown environment, tag not allowed for it, malformed token or login on stdin |

On exit 1 read the output: "rollback failed, the service needs a hand" means neither image
answers and someone has to look at `docker logs`.

## Bootstrap, once per environment

The VM needs Docker 20.10 or later (the unit uses `docker run --pull=never`), `openssl`, `curl`
and `flock`, and the deploy user (`ubuntu`) must be in the
`docker` group and have passwordless `sudo`: `meetween-deploy` runs `docker` as that user and
calls `sudo systemctl` and `sudo sed` on the env file.

```sh
scp -r deploy <vm-user>@<vm-host>:meetween-deploy-files
ssh <vm-user>@<vm-host>
cd meetween-deploy-files
./bootstrap.sh dev
./bootstrap.sh prod
```

It is safe to run again, and should be when a file in `deploy/` changes. It installs:

- `/mnt/vdc/meetween/<env>` and its `recordings`, owned by uid 1000, and a placeholder `image.env`
- `/etc/meetween/<env>.env` from `fastapi.env.example`, `root:root 0600`, with `PORT`,
  `MONGO_DB_NAME`, `RECORDINGS_BASE_PATH` and a new random `JWT_SECRET_KEY` filled in. An existing
  file is never overwritten.
- `/etc/systemd/system/meetween-fastapi@.service`, `/usr/local/bin/meetween-deploy-gate` and
  `/usr/local/bin/meetween-deploy`, then `daemon-reload` and `enable`

It does not start the unit, because `image.env` has no image yet. The first `meetween-deploy`
does. Then edit the values bootstrap leaves at the template:

```sh
sudo vi /etc/meetween/<env>.env
```

- `JITSI_MEET_BASE_URL`: empty, so the token endpoint returns no join URL. Set it to the Jitsi
  web frontend of that environment.
- `MODEL_PROFILE`: `voxtral_mini_3b_2507`. Change it if the environment should use another
  profile.
- Check `MONGO_URI` (`mongodb://localhost:27017`), `ENABLE_AUTH` (`true`) and `LOG_LEVEL`
  (`INFO`); they are usually right as they are.

## CI keys

One key per environment, so the dev key can never touch prod. The GitHub environments in the next
section must exist before the secrets are set. On a trusted machine:

```sh
ssh-keygen -t ed25519 -f meetween-ci-dev -N "" -C meetween-ci-dev
ssh-keygen -t ed25519 -f meetween-ci-prod -N "" -C meetween-ci-prod
```

No passphrase: CI cannot type one. The key is safe because the VM pins it to one command. Append
one line per key to `~ubuntu/.ssh/authorized_keys` on the VM, with the public half:

```
restrict,command="/usr/local/bin/meetween-deploy-gate dev" ssh-ed25519 AAAA... meetween-ci-dev
restrict,command="/usr/local/bin/meetween-deploy-gate prod" ssh-ed25519 AAAA... meetween-ci-prod
```

`restrict` turns off the terminal and every forwarding. The environment comes from the key. The
gate accepts only `status` and `deploy <tag>`, and for prod only `vX.Y.Z`. A deploy may carry
the registry token and login on stdin, as described above.

Store the private halves and the connection details on GitHub:

```sh
gh secret set VM_SSH_KEY --env dev -R Meetween/fastapi-server < meetween-ci-dev
gh secret set VM_SSH_KEY --env prod -R Meetween/fastapi-server < meetween-ci-prod
printf '%s' '<vm-host>' | gh secret set VM_HOST -R Meetween/fastapi-server
ssh-keyscan -t ed25519 <vm-host> 2>/dev/null | gh secret set VM_KNOWN_HOSTS -R Meetween/fastapi-server
gh variable set VM_USER -R Meetween/fastapi-server -b ubuntu
gh variable set DEPLOY_DEV_ENABLED -R Meetween/fastapi-server -b true
```

The host is a secret so it stays masked in the logs of a public repository. `VM_KNOWN_HOSTS`
lets the workflow refuse a host it does not know; compare the fingerprint with the VM's own
before trusting what `ssh-keyscan` returned.

Test each key, then delete the local private keys:

```sh
ssh -i meetween-ci-dev <vm-user>@<vm-host> status       # exit 0 when /health answers
ssh -i meetween-ci-dev <vm-user>@<vm-host> ls           # refused: ls, exit 2
ssh -i meetween-ci-prod <vm-user>@<vm-host> deploy main # refused: bad tag for prod, exit 2
rm meetween-ci-dev meetween-ci-dev.pub meetween-ci-prod meetween-ci-prod.pub
```

`status` answers 1 until the first deploy has run.

## GitHub settings this relies on

- Environments `dev` and `prod`, both with the deployment branch policy `main`. `prod` also has
  required reviewers, the maintainers team. Each holds its own `VM_SSH_KEY`.
- Rulesets on `main`: a pull request is required, the checks `Test` and `Conventional Commits
  title` must pass, and only the maintainers merge.
- The GHCR package `fastapi-server` is private; the organisation does not allow public packages.
  It must stay linked to this repository: a package linked to a repository grants that
  repository's workflows access, so the `GITHUB_TOKEN` of `deploy: vm` with `packages: read` can
  pull it. Check under the package settings, "Manage Actions access", that
  `Meetween/fastapi-server` is listed with read access or more. No registry credential is stored
  on the VM or in the repository secrets.

## Cutover of prod to 8001

Do it with the colleague who runs the current hand-started process, who knows how to start it
again. Until then prod answers on 8012 and the old process keeps 8001.

1. Set `PORT=8001` in `/etc/meetween/prod.env`. `meetween-deploy` reads the port from that file.
2. Stop the old process with SIGINT so uvicorn runs its shutdown hooks:
   `ss -ltnp | grep :8001`, then `kill -INT <pid>`.
3. `sudo systemctl restart meetween-fastapi@prod`
4. `curl -s http://127.0.0.1:8001/health` reports the release version.
5. Check Jigasi and the Jitsi web frontend, which reach the service at `host.docker.internal:8001`:
   open a meeting and watch captions arrive.

Two things change for users. The JWT secret is new, so they log in again. New recordings go to
`/mnt/vdc/meetween/prod/recordings`; what the old process wrote stays where it is and is no
longer cleaned. Keep the old folder for a week.

Rollback: `sudo systemctl stop meetween-fastapi@prod`, set `PORT=8012` again, start the old
process as before.

## MongoDB

It is not deployed from here. One shared instance runs on the VM and both environments use it,
through the databases `meetween_dev` and `meetween`. `docker-compose.yml` in the repository
starts a MongoDB for local development only; the deploy never uses it.
