# Workflows

| File | Name | Triggered by | Purpose |
| --- | --- | --- | --- |
| `ci.yml` | `check: ci` | pull request, push to `main` | runs the test suite |
| `pr-title.yml` | `check: pr title` | pull request opened, edited, reopened, synchronized | validates the title against Conventional Commits |
| `release.yml` | `release: draft or cut` | push to `main` | keeps the release pull request current, cuts the release when it is merged, dispatches the deploy at the tag |
| `deploy.yml` | `deploy: vm` | manual dispatch with a ref, or `release.yml` at a tag | deploys one ref to the VM over SSH |

## Naming

Every workflow is named `category: detail`. The Actions sidebar sorts alphabetically and has
no grouping of its own, so the prefix does the grouping.

- `check:` runs on a pull request and reports on the merge
- `release:` cuts versions and tags
- `deploy:` ships a ref to an environment

## Timeouts

Every job sets `timeout-minutes: 15`. The GitHub default is 360 minutes, which turns a hung
step into six hours of occupied runner.

## Pinning

Actions are referenced by commit SHA with the version in a trailing comment. Dependabot keeps
both current.
