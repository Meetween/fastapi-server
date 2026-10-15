# Workflows

| File | Name | Triggered by | Purpose |
| --- | --- | --- | --- |
| `ci.yml` | `check: ci` | pull request, push to `main` | runs the test suite |
| `pr-title.yml` | `check: pr title` | pull request opened, edited, reopened, synchronized | validates the title against Conventional Commits |
| `image.yml` | `build: image` | pull request, push to `main`, manual dispatch with a tag or commit | builds the container image; pushes `sha-<commit>` and `main` from `main`, backfills old tags on dispatch |
| `release.yml` | `release: draft or cut` | push to `main` | keeps the release pull request current, cuts the release when it is merged, promotes the image to the tag, dispatches the prod deploy |
| `deploy.yml` | `deploy: vm` | `build: image` completed on `main`, manual dispatch with an environment | deploys an image tag to `dev` or `prod` on the VM over SSH |

## Naming

Every workflow is named `category: detail`. The Actions sidebar sorts alphabetically and has
no grouping of its own, so the prefix does the grouping.

- `check:` runs on a pull request and reports on the merge
- `build:` produces an artifact
- `release:` cuts versions and tags
- `deploy:` ships a ref to an environment

## Timeouts

Every job sets `timeout-minutes: 15`. The GitHub default is 360 minutes, which turns a hung
step into six hours of occupied runner. One exception: the promote job in `release.yml` waits
for `image.yml` to finish its own 15 minute build, so it gets 25.

## Pinning

Actions are referenced by commit SHA with the version in a trailing comment. Dependabot keeps
both current.
