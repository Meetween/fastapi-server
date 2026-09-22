# Contributing

## Setup

Python 3.12, [uv](https://docs.astral.sh/uv/) and ffmpeg.

```sh
uv sync
uv run pytest
uv run main.py
```

## Branches

`main` is the only long lived branch. Branch from it as `feat/<topic>`, `fix/<topic>` or
`chore/<topic>`, keep the branch short lived, open a pull request against `main`.

## Pull requests

The title follows Conventional Commits, `<type>[(scope)][!]: <description>`, and a check
refuses anything else. Merges are squashed: the title becomes the commit subject on `main`
and the description becomes its body, so both are worth writing carefully. Only `feat`,
`fix`, `perf`, `refactor` and `revert` reach the changelog; `chore`, `ci`, `docs`, `test`,
`build` and `style` are silent.

Two checks run on every pull request: the title check and the test suite. Neither is
enforced by GitHub on the current plan, so a red check is a reason not to merge, not a lock.

Do not edit `CHANGELOG.md`. It is generated.

An optional local aid for commit messages is [commitizen](https://commitizen-tools.github.io/commitizen/):
`uv tool install commitizen`, then `cz commit`.

## Releases

release-please keeps one pull request open, `chore(main): release x.y.z`, and rewrites it on
every merge. Merge it to release: the tag `vX.Y.Z`, the GitHub Release and the deployment to
the VM follow on their own. Versions are `0.x`: `feat` bumps the minor, `fix` bumps the patch.

To force a version, add a line `Release-As: X.Y.Z` to the description of a `feat` or `fix`
pull request before merging it. On a hidden type the line is ignored.

To deploy a branch to the VM before it is released: `gh workflow run deploy.yml -f ref=<branch>`.
See `deploy/README.md`.
